import numpy, h5py, IPython,astropy,copy,sys,time,subprocess
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9
import _pickle as pickle
import os.path
import Misc.in1d_parallel as in1d_parallel
from matplotlib.colors import LogNorm

class TransportObject: pass

def CalcT(u,nume):
    "u:GasAttrs['InternalEnergy'], nume:GasAttrs['ElectronAbundance']."
    g_gamma= 5.0/3.0
    g_minus_1= g_gamma-1.0
    PROTONMASS = 1.6726e-24
    BoltzMann_ergs= 1.3806e-16
    BoltzMann_eV= 8.617e-5
    UnitMass_in_g  = 1.989e43 #  1.0e10 solar masses
    UnitEnergy_in_cgs= 1.989e53

    XH= 0.76
    yhelium= (1-XH)/(4*XH)
    mu= (1+4*yhelium)/(1+yhelium+nume)
    MeanWeight= mu*PROTONMASS
    Temp = MeanWeight/BoltzMann_ergs * g_minus_1 * u * 1e+10
    print(mu)
    return Temp


def CalcMu(nume):
    "nume:GasAttrs['ElectronAbundance']."
    g_gamma= 5.0/3.0
    g_minus_1= g_gamma-1.0
    PROTONMASS = 1.6726e-24
    BoltzMann_ergs= 1.3806e-16
    BoltzMann_eV= 8.617e-5
    UnitMass_in_g  = 1.989e43 #  1.0e10 solar masses
    UnitEnergy_in_cgs= 1.989e53

    XH= 0.76
    yhelium= (1-XH)/(4*XH)
    mu= (1+4*yhelium)/(1+yhelium+nume)

    return mu



def SavePickle(PickleFile,Object):
    f=open(PickleFile, 'wb')
    pickle.dump(Object,f)
    f.close()

def LoadPickle(PickleFile):
    f=open(PickleFile, 'rb')
    Object = pickle.load(f)
    f.close()
    return Object


def LoadLookbackTimePickle():
    if os.path.exists('LookbackTime_WMAP9.pickle') == False:
        CreatePickle()

    return LoadPickle('LookbackTime_WMAP9.pickle')

def CreatePickle():
    a = numpy.arange(0.01,1.02,0.0005)
#    Cosmo = FlatLambdaCDM(H0=70, Om0=0.27)
    #Cosmo = FlatLambdaCDM(H0=71, Om0=0.266)
    Cosmo=WMAP9
    t_lookback = Cosmo.lookback_time(1.0/a - 1.0)

    StarFormLookback = numpy.interp([0.5],a,t_lookback)
    T = TransportObject()
    T.t_lookback = t_lookback
    T.a = a
    T.command = 'StarFormLookback = numpy.interp([0.5],a,t_lookback)'
    SavePickle('LookbackTime_WMAP9.pickle',T)




class TrackGalaxy:
    #def __init__(self, SnapshotNumbers, SimName, Dir = '/home/ms/Desktop/MergerSim/1526/DeBruyne/ZoomFactor1Bubble/',SnapBase = 'snapshot',GroupBase = 'fof_subhalo_tab',TreeBase = 'trees_sf1',TreeDir = './',MpcInSnap=False,MultipleSnaps=False):
    def __init__(self, SnapshotNumbers, SimName, Dir = '/store/erebos/sparre/Mergers/1330DeBruyne_ZoomFactor3/output/',SnapBase = 'snapshot',GroupBase = 'fof_subhalo_tab',TreeBase = 'trees_sf1',TreeDir = './',MpcInSnap=False,MultipleSnaps=False):
    #def __init__(self, SnapshotNumbers, SimName, Dir = '/store/erebos/sparre/Mergers/1330DeBruyne_ZoomFactor3/output/',SnapBase = 'snapshot',GroupBase = 'fof_subhalo_tab',TreeBase = 'trees_sf1',AnalysisModules=[],TreeDir = './',MpcInSnap=False,MultipleSnaps=False):
        self.SimName = SimName
        self.Snapshots = SnapshotNumbers
        self.SnapTimes = []
        self.Dir = Dir
        self.SnapBase = SnapBase
        self.GroupBase = GroupBase
        self.TreeBase = TreeBase
        self.TreeDir = TreeDir
        self.MpcInSnap = MpcInSnap
        self.MultipleSnaps = MultipleSnaps #flag to set whether multiple files are used in snapshots

        self.Filenames = []
        for i in range(len(self.Snapshots)):
            if MultipleSnaps==False:
                self.Filenames.append(self.Dir+self.SnapBase+'_'+str(self.Snapshots[i]).zfill(3)+'.hdf5')
            else:
                self.Filenames.append(self.Dir+'/snapdir_'+str(self.Snapshots[i]).zfill(3)+'/'+self.SnapBase+'_'+str(self.Snapshots[i]).zfill(3)+'.0.hdf5')


            try:
                Snap = h5py.File(self.Filenames[i],'r')
            except IOError:
                    print( 'File ', self.Filenames[i], 'Doesnt exists....')
                    sys.exit()

            self.SnapTimes.append(Snap['Header'].attrs.get('Time'))
            Snap.close()
        self.SnapTimes = numpy.array(self.SnapTimes)

        self.LookbackTime =WMAP9.lookback_time(1.0/self.SnapTimes - 1).value


    def CalcSubhaloNumberOfStarParticles( self , snap ):
        #Thanks to Sebastian B for helping with this function!

        Attributes = self.GetParticles(snap,Attrs=['ParticleIDs'],Type=4)
        IDs = Attributes['ParticleIDs']

        Groups = self.GetGroups(snap, Attrs=['/Subhalo/SubhaloLenType', '/Subhalo/SubhaloMass', 'Group/GroupLenType', 'Group/GroupNsubs'])

        GroupLenType = Groups['Group/GroupLenType']
        GroupNsubs = Groups['Group/GroupNsubs']
        SubhaloLenType = Groups['/Subhalo/SubhaloLenType']

        ParticleGroup = numpy.repeat(numpy.arange(GroupLenType.shape[0]),GroupLenType[:,4])
        ParticleGroup = numpy.append( ParticleGroup, numpy.zeros( IDs.shape[0]-numpy.sum(GroupLenType[:,4])  )-1)
        ParticleGroup = numpy.array(ParticleGroup,dtype=numpy.int32)

        ParticleSubhalo = numpy.zeros(IDs.shape[0],dtype=numpy.int32)-1

        ParticleCount = 0
        for i in range(ParticleGroup.max()):
            FirstSubhaloInGroup = numpy.sum(GroupNsubs[0:i])
            LastSubhaloInGroup = FirstSubhaloInGroup +GroupNsubs[i]

            AAA = numpy.zeros(GroupLenType[i,4]) - 1

            tmp = numpy.repeat(numpy.sum(GroupNsubs[0:i])+numpy.arange(GroupNsubs[i]),SubhaloLenType[FirstSubhaloInGroup:LastSubhaloInGroup,4])
            AAA[0:tmp.shape[0]] = tmp
            AAA = numpy.array(AAA,dtype=numpy.int32)


            ParticleSubhalo[ParticleCount:ParticleCount+AAA.shape[0]] = AAA
            ParticleCount += GroupLenType[i,4]

        ParticleSubhalo = numpy.array(ParticleSubhalo,dtype=numpy.int32)

        return ParticleGroup, ParticleSubhalo



    def CalcSubhaloNumberOfParticles( self , snap ,Type=4):

        Attributes = self.GetParticles(snap,Attrs=['ParticleIDs'],Type=Type)
        IDs = Attributes['ParticleIDs']

        Groups = self.GetGroups(snap, Attrs=['/Subhalo/SubhaloLenType', '/Subhalo/SubhaloMass', 'Group/GroupLenType', 'Group/GroupNsubs'])

        GroupLenType = Groups['Group/GroupLenType']
        GroupNsubs = Groups['Group/GroupNsubs']
        SubhaloLenType = Groups['/Subhalo/SubhaloLenType']

        ParticleGroup = numpy.repeat(numpy.arange(GroupLenType.shape[0]),GroupLenType[:,Type])
        ParticleGroup = numpy.append( ParticleGroup, numpy.zeros( IDs.shape[0]-numpy.sum(GroupLenType[:,Type])  )-1)
        ParticleGroup = numpy.array(ParticleGroup,dtype=numpy.int32)

        ParticleSubhalo = numpy.zeros(IDs.shape[0],dtype=numpy.int32)-1

        ParticleCount = 0
        for i in range(ParticleGroup.max()):
            FirstSubhaloInGroup = numpy.sum(GroupNsubs[0:i])
            LastSubhaloInGroup = FirstSubhaloInGroup +GroupNsubs[i]

            AAA = numpy.zeros(GroupLenType[i,Type]) - 1

            tmp = numpy.repeat(numpy.sum(GroupNsubs[0:i])+numpy.arange(GroupNsubs[i]),SubhaloLenType[FirstSubhaloInGroup:LastSubhaloInGroup,Type])
            AAA[0:tmp.shape[0]] = tmp
            AAA = numpy.array(AAA,dtype=numpy.int32)


            ParticleSubhalo[ParticleCount:ParticleCount+AAA.shape[0]] = AAA
            ParticleCount += GroupLenType[i,Type]

        ParticleSubhalo = numpy.array(ParticleSubhalo,dtype=numpy.int32)

        return ParticleGroup, ParticleSubhalo



    def CalcSubhaloNumberOfDMParticles( self , snap ):

        Attributes = self.GetParticles(snap,Attrs=['ParticleIDs'],Type=1)
        IDs = Attributes['ParticleIDs']

        Groups = self.GetGroups(snap, Attrs=['/Subhalo/SubhaloLenType', '/Subhalo/SubhaloMass', 'Group/GroupLenType', 'Group/GroupNsubs'])

        GroupLenType = Groups['Group/GroupLenType']
        GroupNsubs = Groups['Group/GroupNsubs']
        SubhaloLenType = Groups['/Subhalo/SubhaloLenType']

        ParticleGroup = numpy.repeat(numpy.arange(GroupLenType.shape[0]),GroupLenType[:,1])
        ParticleGroup = numpy.append( ParticleGroup, numpy.zeros( IDs.shape[0]-numpy.sum(GroupLenType[:,1])  )-1)
        ParticleGroup = numpy.array(ParticleGroup,dtype=numpy.int32)

        ParticleSubhalo = numpy.zeros(IDs.shape[0],dtype=numpy.int32)-1

        ParticleCount = 0
        for i in range(ParticleGroup.max()):
            FirstSubhaloInGroup = numpy.sum(GroupNsubs[0:i])
            LastSubhaloInGroup = FirstSubhaloInGroup +GroupNsubs[i]

            AAA = numpy.zeros(GroupLenType[i,1]) - 1

            tmp = numpy.repeat(numpy.sum(GroupNsubs[0:i])+numpy.arange(GroupNsubs[i]),SubhaloLenType[FirstSubhaloInGroup:LastSubhaloInGroup,1])
            AAA[0:tmp.shape[0]] = tmp
            AAA = numpy.array(AAA,dtype=numpy.int32)


            ParticleSubhalo[ParticleCount:ParticleCount+AAA.shape[0]] = AAA
            ParticleCount += GroupLenType[i,1]

        ParticleSubhalo = numpy.array(ParticleSubhalo,dtype=numpy.int32)

        return ParticleGroup, ParticleSubhalo


    def GetSubhaloParticles(self,SnapNo,SubhaloNum ,Type = 4,Attrs = ['Coordinates','ParticleIDs','GFM_StellarFormationTime']):
            ThisSubhaloLenType = self.GetGroups(SnapNo=SnapNo)['/Subhalo/SubhaloLenType']
            NStars = ThisSubhaloLenType[:,Type]
            NstarsStart = numpy.sum(NStars[0:SubhaloNum])
            NstarsEnd = numpy.sum(NStars[0:SubhaloNum+1])

            return self.GetParticles(SnapNo=SnapNo,Type=Type,Attrs = Attrs,SubhaloIDs=[NstarsStart,NstarsEnd])


    def CalcSFR(self,SnapNo,SubhaloNum=0):
        Attrs = self.GetSubhaloParticles(SnapNo,SubhaloNum,Attrs = ['GFM_InitialMass','GFM_StellarFormationTime'],Type=4)
        M = Attrs['GFM_InitialMass']
        Time = Attrs['GFM_StellarFormationTime']
        GoodIDs = numpy.where(Time>0)
        M = M[GoodIDs]*1e10/0.7
        Time = Time[GoodIDs]

        a = numpy.arange(0.01,1.02,0.001)
        t_lookback = WMAP9.lookback_time(1.0/a - 1)
        StarFormLookback = numpy.interp(Time,a,t_lookback)

        y,x=numpy.histogram( StarFormLookback, weights=M,bins=500,range=(0.0,14.0))
        dx = x[1] - x[0]
        x = x[0:-1] + 0.5 * ( x[1] - x[0] )
        SFR = y / (x[1]-x[0]) / 1e9
        return x,SFR


    def GetParticles(self,SnapNo = 136,Type = 4,Attrs = ['Coordinates','ParticleIDs','GFM_StellarFormationTime'],SubhaloIDs = None):
        if self.MultipleSnaps == False:
            Attr = {}

            for key in Attrs:
                Attr[key] = []

            Filename = self.Dir+self.SnapBase+'_'+str(SnapNo).zfill(3)+'.hdf5'

            try:
                SnapshotFile = h5py.File(Filename,'r')
            except IOError:
                print( 'File ',Filename, 'Doesnt exists....')

            if SubhaloIDs == None:
                for i in range(len(Attrs)):
                    Attr[Attrs[i]] = SnapshotFile['/PartType'+str(Type)+'/'+Attrs[i]][()]#.value
            else:
                for i in range(len(Attrs)):
#                    Attr[Attrs[i]] = SnapshotFile['/PartType'+str(Type)+'/'+Attrs[i]][SubhaloIDs[0]:SubhaloIDs[1]]
                    Attr[Attrs[i]] = SnapshotFile['/PartType'+str(Type)+'/'+Attrs[i]][SubhaloIDs[0]:SubhaloIDs[1]]

            SnapshotFile.close()

            return Attr
        else:
            Attr = {}

            for key in Attrs:
                Attr[key] = []


            #Nfilespersnapshot = len(commands.getoutput('ls -1 '+self.Dir+'/snapdir_'+str(SnapNo).zfill(3)+'/'+'/*.hdf5').split('\n'))
            #len(subprocess.call('ls -1 '+self.Dir+'/groups_'+str(SnapNo).zfill(3)+'/'+'/*.hdf5').split('\n'))
            j=0
            Filename = self.Dir+'/snapdir_'+str(SnapNo).zfill(3)+'/'+self.SnapBase+'_'+str(SnapNo).zfill(3)+'.'+str(j)+'.hdf5'
            try:
                SnapshotFile = h5py.File(Filename,'r')
            except IOError:
                print( 'File ',Filename, 'Doesnt exists....')

            Nfilespersnapshot = SnapshotFile['Header'].attrs.get('NumFilesPerSnapshot')

            for j in range(Nfilespersnapshot):

                Filename = self.Dir+'/snapdir_'+str(SnapNo).zfill(3)+'/'+self.SnapBase+'_'+str(SnapNo).zfill(3)+'.'+str(j)+'.hdf5'

                try:
                    SnapshotFile = h5py.File(Filename,'r')
                except IOError:
                    print( 'File ',Filename, 'Doesnt exists....')

                for i in range(len(Attrs)):
                        if j==0:
                            Attr[Attrs[i]] = SnapshotFile['/PartType'+str(Type)+'/'+Attrs[i]][()]#.value
                        else:
#                            Attr[Attrs[i]] = numpy.append( Attr[Attrs[i]]  ,SnapshotFile['/PartType'+str(Type)+'/'+Attrs[i]].value,axis=0)
                            Attr[Attrs[i]] = numpy.append( Attr[Attrs[i]]  ,SnapshotFile['/PartType'+str(Type)+'/'+Attrs[i]][()],axis=0)
                SnapshotFile.close()

            if SubhaloIDs != None:
                for i in range(len(Attrs)):
                    Attr[Attrs[i]] = Attr[Attrs[i]][SubhaloIDs[0]:SubhaloIDs[1]]

            return Attr




    def DefineSubhaloToTrack_UseBlackHoles(self,SubhaloNum,SnapNo):
        Attrs = self.GetGroups(SnapNo = SnapNo,Attrs = ['/Subhalo/SubhaloLenType'])
        SubhaloLenType = Attrs['/Subhalo/SubhaloLenType']

        NStars = SubhaloLenType[:,5]

        NstarsStart = numpy.sum(NStars[0:SubhaloNum])
        NstarsEnd = numpy.sum(NStars[0:SubhaloNum+1])

        Attrs = self.GetParticles(SnapNo=SnapNo,Type = 5,Attrs = ['ParticleIDs','Coordinates'],SubhaloIDs = [NstarsStart,NstarsEnd])
        IDs = Attrs['ParticleIDs']

        self.SubhaloNumberOfTrackedHalo = self.Snapshots*0-9999
        self.GroupNumberOfTrackedHalo = self.Snapshots*0-9999

        self.SubhaloToTrack_SubhaloNum = SubhaloNum
        self.SubhaloToTrack_Snapshot = SnapNo
        self.StellarIDs = IDs

        self.IDs_LastSnap = copy.deepcopy(IDs)



    def DefineSubhaloToTrack(self,SubhaloNum,SnapNo,DownSampleN=None,DownSampleN_MostBound = False):
        Attrs = self.GetGroups(SnapNo = SnapNo,Attrs = ['/Subhalo/SubhaloLenType'])
        SubhaloLenType = Attrs['/Subhalo/SubhaloLenType']

        NStars = SubhaloLenType[:,4]

        NstarsStart = numpy.sum(NStars[0:SubhaloNum])
        NstarsEnd = numpy.sum(NStars[0:SubhaloNum+1])

        Attrs = self.GetParticles(SnapNo=SnapNo,Type = 4,Attrs = ['ParticleIDs','GFM_StellarFormationTime'],SubhaloIDs = [NstarsStart,NstarsEnd])
        IDs = Attrs['ParticleIDs']
        FormTime = Attrs['GFM_StellarFormationTime']
        StarIDs = numpy.where(Attrs['GFM_StellarFormationTime']>0.0)
        IDs = IDs[StarIDs]
        FormTime = FormTime[StarIDs]

        Argsort = numpy.argsort(IDs)
        IDs = IDs[Argsort]
        FormTime = FormTime[Argsort]

        if DownSampleN != None and IDs.shape[0] > DownSampleN:
            if DownSampleN_MostBound == False:
                RandInts = numpy.random.randint(IDs.shape[0],size=DownSampleN)
                RandInts = numpy.unique(RandInts)
            else:
                RandInts = numpy.arange(DownSampleN)
                #RandInts = numpy.argsort(FormTime)[::-1][0:DownSampleN]#Select youngest stars only
                #RandInts = numpy.argsort(FormTime)[0:DownSampleN]#Select oldest stars only
            IDs = IDs[RandInts]
            FormTime = FormTime[RandInts]

        self.SubhaloNumberOfTrackedHalo = self.Snapshots*0-9999
        self.GroupNumberOfTrackedHalo = self.Snapshots*0-9999

        self.SubhaloToTrack_SubhaloNum = SubhaloNum
        self.SubhaloToTrack_Snapshot = SnapNo
        self.StellarIDs = IDs
        self.StellarFormTime = FormTime

        self.IDs_LastSnap = copy.deepcopy(IDs)



    def GetGroups(self,SnapNo = 136,Attrs = ['/Subhalo/SubhaloLenType','/Subhalo/SubhaloMass']):

        if self.MultipleSnaps == False:
            Attr = {}

            for key in Attrs:
                Attr[key] = []

            Filename = self.Dir+self.GroupBase+'_'+str(SnapNo).zfill(3)+'.hdf5'

            try:
                GroupFile = h5py.File(Filename,'r')
            except IOError:
                print( 'File ',Filename, 'Doesnt exists....' )

            for i in range(len(Attrs)):
                Attr[Attrs[i]] = GroupFile[Attrs[i]][()]#.value

            GroupFile.close()

            return Attr
        elif self.MultipleSnaps == True:
            Attr = {}

            for key in Attrs:
                Attr[key] = []

            #Nfilespersnapshot = len(commands.getoutput('ls -1 '+self.Dir+'/groups_'+str(SnapNo).zfill(3)+'/'+'/*.hdf5').split('\n'))
            #len(subprocess.call('ls -1 '+self.Dir+'/groups_'+str(SnapNo).zfill(3)+'/'+'/*.hdf5').split('\n'))

            j=0
            Filename = self.Dir+'/snapdir_'+str(SnapNo).zfill(3)+'/'+self.SnapBase+'_'+str(SnapNo).zfill(3)+'.'+str(j)+'.hdf5'
            try:
                SnapshotFile = h5py.File(Filename,'r')
            except IOError:
                print( 'File ',Filename, 'Doesnt exists....')

            Nfilespersnapshot = SnapshotFile['Header'].attrs.get('NumFilesPerSnapshot')

            for j in range(Nfilespersnapshot):
                Filename = self.Dir+'/groups_'+str(SnapNo).zfill(3)+'/'+self.GroupBase+'_'+str(SnapNo).zfill(3)+'.'+str(j)+'.hdf5'

                try:
                    GroupFile = h5py.File(Filename,'r')
                except IOError:
                    print( 'File ',Filename, 'Doesnt exists....')

                for i in range(len(Attrs)):
                    if j==0:

                        Attr[Attrs[i]] = GroupFile[Attrs[i]][()]#GroupFile[Attrs[i]].value
                    else:
                        if Attrs[i] in GroupFile:
                            Attr[Attrs[i]] = numpy.append( Attr[Attrs[i]]  ,  GroupFile[Attrs[i]][()] ,axis=0)

                GroupFile.close()
            return Attr


    def TrackProgenitor(self,SubhaloAttrsToSave=['Subhalo/SubhaloSFR','Subhalo/SubhaloBHMdot','Subhalo/SubhaloBHMass'],GroupAttrsToSave = ['Group/Group_M_Crit200','Group/Group_R_Crit200'],UpdateIDsAtEachStep = False,CreateImage=False,CreateGasImage=False,AnalysisModules = [],PickleName = None,HardCodedSnaps=[],HardCodedSubhalos=[],HardCodedGroups=[]):

        GalaxyProp = {}
        for key in SubhaloAttrsToSave:
            GalaxyProp[key] = []

        for key in GroupAttrsToSave:
            GalaxyProp[key] = []

        self.Modules = []
        for module in AnalysisModules:
            self.Modules.append(module())

        if PickleName != None and os.path.exists(PickleName):
            print( 'Reading ', PickleName, 'instead of calculating progenitor branch again.' )
            T = LoadPickle(PickleName)
            self.GalaxyProp = T.GalaxyProp
            self.SubhaloNumberOfTrackedHalo   =  T.SubhaloNumberOfTrackedHalo
            self.GroupNumberOfTrackedHalo   =  T.GroupNumberOfTrackedHalo

            if len(self.LookbackTime) != len(T.LookbackTime):
                print( 'Something is completely wrong!!!len(self.LookbackTime) != len(T.LookbackTime)' )
                sys.exit()
            return None

        for i in range(len(self.Snapshots)):
            t0 = time.time()
            ThisSnapNo = self.Snapshots[i]

            print( 'Now at Snapshot ',ThisSnapNo)
            if ThisSnapNo in HardCodedSnaps:
                ThisIndex = HardCodedSnaps.index(ThisSnapNo)

                print( 'Hardcoding snapshot %d to Subhalo %d in Group %d.'%(ThisSnapNo,HardCodedSubhalos[ThisIndex],HardCodedGroups[ThisIndex]) )
                self.SubhaloNumberOfTrackedHalo[i] = HardCodedSubhalos[ThisIndex]
                self.GroupNumberOfTrackedHalo[i] = HardCodedGroups[ThisIndex]
                SubhaloNumber = HardCodedSubhalos[ThisIndex]
                GroupNumber = HardCodedGroups[ThisIndex]
            else:
                IDsInSnap = self.GetParticles(SnapNo = ThisSnapNo, Attrs = ['ParticleIDs'])['ParticleIDs']

                #tAA = time.time()
                ParticleGroup, ParticleSubhalo = self.CalcSubhaloNumberOfStarParticles(ThisSnapNo)
                #print 'Time used in new subhalo number calc func:',time.time()-tAA

                FromLastProgenitor = in1d_parallel.in1d_parallel(IDsInSnap,self.IDs_LastSnap ,Ncpu=12)

                t2 = time.time()

                tmp = ParticleSubhalo[FromLastProgenitor] + 1#creating offset to make compatible with numpy.bincount
                BinCount = numpy.bincount(tmp)
                SubhaloNumber =  numpy.argmax(BinCount) - 1

                self.SubhaloNumberOfTrackedHalo[i] = SubhaloNumber

                #save GrNr here:
                GroupNumber = self.GetGroups(SnapNo = ThisSnapNo,Attrs = ['Subhalo/SubhaloGrNr'])['Subhalo/SubhaloGrNr'][SubhaloNumber]
                self.GroupNumberOfTrackedHalo[i] = GroupNumber

            #Save stuff
            Attrs = self.GetGroups(SnapNo = ThisSnapNo,Attrs = SubhaloAttrsToSave)

            for key in SubhaloAttrsToSave:
                GalaxyProp[key].append(Attrs[key][SubhaloNumber])

            Attrs = self.GetGroups(SnapNo = ThisSnapNo,Attrs = GroupAttrsToSave)

            for key in GroupAttrsToSave:
                GalaxyProp[key].append(Attrs[key][GroupNumber])

            for module in self.Modules:
                module(self,i,SubhaloNumber)

            #UpdateIDs:
            if UpdateIDsAtEachStep:
                Attrs = self.GetSubhaloParticles(ThisSnapNo,SubhaloNumber,Type = 4,Attrs = ['ParticleIDs','GFM_StellarFormationTime'])
                #self.IDs_LastSnap = self.GetParticles(SnapNo = ThisSnapNo,Type = 4,Attrs = [ParticleIDs],SubhaloIDs = None)
                NewIDs = Attrs['ParticleIDs'][numpy.where(Attrs['GFM_StellarFormationTime']>0.0)]

                if NewIDs.shape[0]>10000:
                    NewIDs = NewIDs[0:10000]

                NewIDs = NewIDs[numpy.argsort(NewIDs)]
                #self.IDs_LastSnap = self.IDs_LastSnap
                IPython.embed()
                self.IDs_LastSnap = NewIDs+0

            if CreateImage == True:
                self.CreateImage(ThisSnapNo,SubhaloNumber)

            if CreateGasImage == True:
                self.CreateGasImage(ThisSnapNo,SubhaloNumber)

        for key in SubhaloAttrsToSave:
            GalaxyProp[key]  = numpy.array(GalaxyProp[key])

        for key in GroupAttrsToSave:
            GalaxyProp[key]  = numpy.array(GalaxyProp[key])

        for module in self.Modules:
            module.Finish()
            for key in module.Attrs.keys():
                GalaxyProp[key] = module.Attrs[key]

        self.GalaxyProp = GalaxyProp
        self.SnapTimes
        self.LookbackTime
        self.Snapshots

        if PickleName != None:
            T = TransportObject()
            T.GalaxyProp = GalaxyProp
            T.SnapTimes = self.SnapTimes
            T.LookbackTime = self.LookbackTime
            T.Snapshots = self.Snapshots
            T.SubhaloNumberOfTrackedHalo = self.SubhaloNumberOfTrackedHalo
            T.GroupNumberOfTrackedHalo = self.GroupNumberOfTrackedHalo
            SavePickle(PickleName,T)


    def TrackProgenitor_UseBlackHoles(self,SubhaloAttrsToSave=['Subhalo/SubhaloSFR','Subhalo/SubhaloBHMdot','Subhalo/SubhaloBHMass'],GroupAttrsToSave = ['Group/Group_M_Crit200','Group/Group_R_Crit200'],UpdateIDsAtEachStep = False,CreateImage=False,CreateGasImage=False,AnalysisModules = [],PickleName = None,HardcodedGroup=None,HardcodedSubhalo=None):

        GalaxyProp = {}
        for key in SubhaloAttrsToSave:
            GalaxyProp[key] = []

        for key in GroupAttrsToSave:
            GalaxyProp[key] = []

        self.Modules = []
        for module in AnalysisModules:
            self.Modules.append(module())

        if PickleName != None and os.path.exists(PickleName):
            print( 'Reading ', PickleName, 'instead of calculating progenitor branch again.')
            T = LoadPickle(PickleName)
            self.GalaxyProp = T.GalaxyProp
            self.SubhaloNumberOfTrackedHalo   =  T.SubhaloNumberOfTrackedHalo
            self.GroupNumberOfTrackedHalo   =  T.GroupNumberOfTrackedHalo

            if len(self.LookbackTime) != len(T.LookbackTime):
                print( 'Something is completely wrong!!!len(self.LookbackTime) != len(T.LookbackTime)' )
                sys.exit()
            return None

        for i in range(len(self.Snapshots)):
            t0 = time.time()
            ThisSnapNo = self.Snapshots[i]


            print( 'Now at Snapshot ',ThisSnapNo)
            if HardcodedGroup!=None and HardcodedSubhalo != None:
                self.SubhaloNumberOfTrackedHalo[i] = HardcodedSubhalo[i]
                self.GroupNumberOfTrackedHalo[i] = HardcodedGroup[i]
            else:
                IDsInSnap = self.GetParticles(SnapNo = ThisSnapNo,Type=5, Attrs = ['ParticleIDs'])['ParticleIDs']

                ParticleGroup, ParticleSubhalo = self.CalcSubhaloNumberOfParticles(ThisSnapNo,Type=5)

                t1 = time.time()

                #FromLastProgenitorOld = numpy.in1d(IDsInSnap,self.IDs_LastSnap , assume_unique = True)
                FromLastProgenitor = in1d_parallel.in1d_parallel(IDsInSnap,self.IDs_LastSnap ,Ncpu=2)
                IPython.embed()
                t2 = time.time()

                tmp = ParticleSubhalo[FromLastProgenitor] + 1#creating offset to make compatible with numpy.bincount
                BinCount = numpy.bincount(tmp)
                SubhaloNumber =  numpy.argmax(BinCount) - 1

                self.SubhaloNumberOfTrackedHalo[i] = SubhaloNumber
                print( 'Determined SubhaloNumber to be %d at snapshot %d '%(SubhaloNumber,ThisSnapNo) )

                #save GrNr here:
                GroupNumber = self.GetGroups(SnapNo = ThisSnapNo,Attrs = ['Subhalo/SubhaloGrNr'])['Subhalo/SubhaloGrNr'][SubhaloNumber]
                self.GroupNumberOfTrackedHalo[i] = GroupNumber


            #Save stuff
            Attrs = self.GetGroups(SnapNo = ThisSnapNo,Attrs = SubhaloAttrsToSave)

            for key in SubhaloAttrsToSave:
                GalaxyProp[key].append(Attrs[key][SubhaloNumber])

            Attrs = self.GetGroups(SnapNo = ThisSnapNo,Attrs = GroupAttrsToSave)

            for key in GroupAttrsToSave:
                GalaxyProp[key].append(Attrs[key][GroupNumber])

            for module in self.Modules:
                module(self,i,SubhaloNumber)

            t3 = time.time()


        for key in SubhaloAttrsToSave:
            GalaxyProp[key]  = numpy.array(GalaxyProp[key])

        for key in GroupAttrsToSave:
            GalaxyProp[key]  = numpy.array(GalaxyProp[key])

        for module in self.Modules:
            module.Finish()
            for key in module.Attrs.keys():
                GalaxyProp[key] = module.Attrs[key]

        self.GalaxyProp = GalaxyProp
        self.SnapTimes
        self.LookbackTime
        self.Snapshots

        if PickleName != None:
            T = TransportObject()
            T.GalaxyProp = GalaxyProp
            T.SnapTimes = self.SnapTimes
            T.LookbackTime = self.LookbackTime
            T.Snapshots = self.Snapshots
            T.SubhaloNumberOfTrackedHalo = self.SubhaloNumberOfTrackedHalo
            T.GroupNumberOfTrackedHalo = self.GroupNumberOfTrackedHalo
            SavePickle(PickleName,T)
