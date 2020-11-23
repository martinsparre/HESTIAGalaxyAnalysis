#/usr/bin/python
#-*- coding: utf-8 -*-
import sys,time
import matplotlib
matplotlib.use('Agg') #Enable this when running from Cluster account. This disables matplotlib.show() command.
import numpy, IPython
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import spatial
import TrackGalaxy
import astropy.units as u
import astropy.constants as const
import MaanIonisationTables.mhh_ion_tables as mhh_ion_tables
import Spectra
import Misc.Papaya


###############################################################
###################### Reading in snapshot ####################
###############################################################

h=0.7
SimName = 'Hestia09-18'
SnapNo = 127
RMAX = 1400.0#An arbitrary number

if SimName == 'Hestia17-11':
    SubhaloNumberMW = 1#These numbers come from cross-correlating with /z/nil/codes/HESTIA/FIND_LG/LGs_8192_GAL_FOR.txt andArepo's SUBFIND.
    SubhaloNumberM31 = 0
    SimulationDirectory = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/17_11/output_2x2.5Mpc/'
elif SimName == 'Hestia09-18':
    SubhaloNumberMW = 3911
    SubhaloNumberM31 = 2608
    SimulationDirectory = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/'
elif SimName == 'Hestia37-11':
    SubhaloNumberMW = 920
    SubhaloNumberM31 = 0
    SimulationDirectory = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/37_11/output_2x2.5Mpc/'
else:
    print( 'SimName',SimName,'is not properly set, try again!. exiting.')
    sys.exit()
    

print( 'Initialising TrackGalaxy class')
T = TrackGalaxy.TrackGalaxy(numpy.array([SnapNo]),SimName,Dir = SimulationDirectory,MultipleSnaps=True) #Imports TrackGalaxy module from TrackGalaxy script
SnapTime = T.SnapTimes[0]#SnapTime is the scale factor
Redshift = 1.0/SnapTime-1


##############################################################
################## We read in group catalog ##################
##############################################################

#Read in position, SFR, stellar mass, gas mass, dark matter mass of all the galaxies (and subhalos) from the simulation
GroupCatalog = T.GetGroups(SnapNo,Attrs=['/Subhalo/SubhaloPos','/Subhalo/SubhaloSFR','/Subhalo/SubhaloMassType','/Subhalo/SubhaloHalfmassRad','/Subhalo/SubhaloCM', '/Subhalo/SubhaloVel','Group/Group_R_Crit200','Group/GroupVel','/Subhalo/SubhaloGrNr'])

#we get the subhalo center
SubhaloPos = 1000*GroupCatalog['/Subhalo/SubhaloPos']*SnapTime/h#in kpc
SubhaloMstar = GroupCatalog['/Subhalo/SubhaloMassType'][:,4]*1e10/h#in Msun
SubhaloMgas = GroupCatalog['/Subhalo/SubhaloMassType'][:,0]*1e10/h#in Msun
SubhaloGrNr = GroupCatalog['/Subhalo/SubhaloGrNr']
SubhaloVel = GroupCatalog['/Subhalo/SubhaloVel']#km/s
GroupVel = GroupCatalog['Group/GroupVel']/SnapTime#km/s. note, Group/GroupVel has units of km/s/SnapTime in simulations units. See https://www.illustris-project.org/data/docs/specifications/ for details.

#Note: velocities are confusing in Arepo units. In subhalos, groups and gas cells three different units are used. km/s, km/s/Snaptime and km/s*sqrt(SnapTime) ,respectively.
#See https://www.illustris-project.org/data/docs/specifications/ for details.

##############################################################
################## We read in gas cells ######################
##############################################################

print( 'We are now reading in gas cells')
tstartread = time.time()
Gas_Attrs = T.GetParticles(SnapNo,Type = 0,Attrs = ['Coordinates','Masses','StarFormationRate','GFM_Metallicity','Velocities','ElectronAbundance','Density','InternalEnergy','GFM_Metals','ParticleIDs'])
print( 'We finished reading data, it took (sec)',time.time()-tstartread)

GasPos = 1000*Gas_Attrs['Coordinates']*SnapTime/h#kpc
GasSFR = Gas_Attrs['StarFormationRate']#Msun/yr
GasDensity = 1e-9*Gas_Attrs['Density']#system units
GasParticleIDs = Gas_Attrs['ParticleIDs']
GasMass = Gas_Attrs['Masses']#1e10Msun/h
GasMetallicity = Gas_Attrs['GFM_Metallicity']/0.0127#In solar units
GasMetals = Gas_Attrs['GFM_Metals']
GasInternalEnergy = Gas_Attrs['InternalEnergy']#km**2/s**2
GasElectronAbundance = Gas_Attrs['ElectronAbundance']
GasVel = Gas_Attrs['Velocities']*numpy.sqrt(SnapTime)#km/s

GasGroupNumber, GasSubhaloNumber = T.CalcSubhaloNumberOfParticles( SnapNo ,Type=0)#We calculate GroupNumber and SubhaloNumber of each particle - this is sometimes useful to have!

##############################################################
################## Centering of coordinates ##################
##############################################################

CenterinMW = False

#Centering coordinates. Either in MW or midpoint of MW or M31
if CenterinMW:
    GasPos[:,0] -= SubhaloPos[SubhaloNumberMW,0]#Center in the MW
    GasPos[:,1] -= SubhaloPos[SubhaloNumberMW,1]
    GasPos[:,2] -= SubhaloPos[SubhaloNumberMW,2]
else:
    GasPos[:,0] -= 0.5*(SubhaloPos[SubhaloNumberMW,0]+SubhaloPos[SubhaloNumberM31,0])#Center in midpoint of MW and M31
    GasPos[:,1] -= 0.5*(SubhaloPos[SubhaloNumberMW,1]+SubhaloPos[SubhaloNumberM31,1])
    GasPos[:,2] -= 0.5*(SubhaloPos[SubhaloNumberMW,2]+SubhaloPos[SubhaloNumberM31,2])

#We measure velocity relative to the M31's Group
GasVel[:,0] -= GroupVel[SubhaloGrNr[SubhaloNumberM31],0]
GasVel[:,1] -= GroupVel[SubhaloGrNr[SubhaloNumberM31],1]
GasVel[:,2] -= GroupVel[SubhaloGrNr[SubhaloNumberM31],2]

#We cut out gas far away:
index_of_nearby_gas = numpy.where(GasPos[:,0]**2+GasPos[:,1]**2+GasPos[:,2]**2<RMAX**2)

print( 'After making a radial cut of r<',RMAX,'kpc we reduced Nparticles from ',GasMass.shape[0],'to',index_of_nearby_gas[0].shape[0])

GasPos = GasPos[index_of_nearby_gas]
GasDensity = GasDensity[index_of_nearby_gas]
GasParticleIDs = GasParticleIDs[index_of_nearby_gas]
GasSFR = GasSFR[index_of_nearby_gas]
GasMass = GasMass[index_of_nearby_gas]
GasMetallicity = GasMetallicity[index_of_nearby_gas]
GasMetals = GasMetals[index_of_nearby_gas]
GasVel = GasVel[index_of_nearby_gas]
GasElectronAbundance = GasElectronAbundance[index_of_nearby_gas]
GasInternalEnergy = GasInternalEnergy[index_of_nearby_gas]
GasGroupNumber = GasGroupNumber[index_of_nearby_gas]
GasSubhaloNumber = GasSubhaloNumber[index_of_nearby_gas]

GasMassDensity_PhysUnits = GasDensity.astype(numpy.float64)*1.0e10/h*const.M_sun/(SnapTime*u.kpc/h)**3
mu = (2*0.7381+0.75*0.2485+0.0134/2)**(-1)#mean molecular weight

GasNumberDensity = (GasMassDensity_PhysUnits/mu/const.m_p).to(1/u.cm**3).value#in 1/cm**3
GasNumberDensityH = (12.0/27.0)*GasNumberDensity#in 1/cm**3

GasTemp = TrackGalaxy.CalcT(GasInternalEnergy,GasElectronAbundance)#in Kelvin

#do a metallicity floor at 10**-4.5 to avoid values much lower than the minimum value of the cloudy tables:
GasMetallicity[numpy.where(GasMetallicity < 10**-4.5)] = 10**-4.5

###############################################################
#####  Initialise Atom class used for ion calculations ########
###############################################################

#Maan Hani has to be offerend coauthorship, if you use these functions. From https://ui.adsabs.harvard.edu/abs/2019MNRAS.488..135H/abstract and https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.1160H/abstract

#Comment in different lines to do different ions. Here we do Ly alpha / HI.

ThisAtom = Spectra.Atom("Hydrogen",'Ly_a','MaanIonisationTables/Martin200328_h1.hdf5',1/T.SnapTimes[0]-1)        
#ThisAtom = Spectra.Atom("Oxygen",'OVI_1031','MaanIonisationTables/Martin200328_o6.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Oxygen",'OVII_21','MaanIonisationTables/Martin200328_o7.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Oxygen",'OVIII_2','MaanIonisationTables/Martin200328_o8.hdf5',1/T.SnapTimes[0]-1)#Note this absorption line does not exist, so spectrum should not be analysed. Nlos is still correct!
#ThisAtom = Spectra.Atom("Oxygen",'OI_1302','MaanIonisationTables/Martin200328_o1.hdf5',1/T.SnapTimes[0]-1)       
#ThisAtom = Spectra.Atom("Carbon",'CIV_1548','MaanIonisationTables/Martin200328_c4.hdf5',1/T.SnapTimes[0]-1)       
#ThisAtom = Spectra.Atom("Carbon",'CIV_1550','MaanIonisationTables/Martin200328_c4.hdf5',1/T.SnapTimes[0]-1)       
#ThisAtom = Spectra.Atom("Silicon",'SiII_1193','MaanIonisationTables/Martin200328_si2.hdf5',1/T.SnapTimes[0]-1)   
#ThisAtom = Spectra.Atom("Silicon",'SiII_1260','MaanIonisationTables/Martin200328_si2.hdf5',1/T.SnapTimes[0]-1)   
#ThisAtom = Spectra.Atom("Silicon",'SiII_1526','MaanIonisationTables/Martin200328_si2.hdf5',1/T.SnapTimes[0]-1)   
#ThisAtom = Spectra.Atom("Silicon",'SiIII_1206','MaanIonisationTables/Martin200328_si3.hdf5',1/T.SnapTimes[0]-1)  
#ThisAtom = Spectra.Atom("Silicon",'SiIV_1402','MaanIonisationTables/Martin200328_si4.hdf5',1/T.SnapTimes[0]-1)   
#ThisAtom = Spectra.Atom("Iron",'FeII_1144','MaanIonisationTables/Martin200328_fe2.hdf5',1/T.SnapTimes[0]-1)      

###############################################################
############# Now doing ionization modellings #################
###############################################################

print( 'Looking up CLOUDY tables.')

#We use ThisAtom class to calculate e.g. n(HI)
fHI = ThisAtom.DoIonisation(numpy.log10(GasNumberDensityH), numpy.log10(GasTemp),numpy.log10( GasMetallicity ), 0.0*GasSFR, GasSFR)
GasNumberDensityIon = ThisAtom.CalculateIonNumberDensity(GasDensity,GasMetals)
print( 'Finished CLOUDY tables,')

###############################################################
################# Do further analysis below  ##################
###############################################################

#GasNumberDensityIon is n(HI) in 1/cm**3.
#IPython.embed()

#Do analysis here!
