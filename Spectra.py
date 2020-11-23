from math import pi
import numpy
import sys
from Misc import Papaya
import MaanIonisationTables.mhh_ion_tables as mhh_ion_tables
import astropy.units as u
import astropy.constants as const


class Atom:
    def __init__(self,Atom,Linename,IonisationFilename,Redshift):
        """
        An information wrapper class, storing properties of a modelled absorption line.

        Examples:

        A = Spectra.Atom('Hydrogen','Ly_a','MaanIonisationTables/h1.hdf5',-.-)
        AtomInfo = Spectra.Atom("Hydrogen",'Ly_a','MaanIonisationTables/h1.hdf5',  1/T.SnapTimes[0]-1  )
        ThisAtom = Spectra.Atom("Oxygen",'OVI_1031','MaanIonisationTables/o6.hdf5',1/T.SnapTimes[0]-1)
        AtomInfo = Spectra.Atom("Silicon",'SiII_1526','MaanIonisationTables/si2.hdf5',  1/T.SnapTimes[0]-1  )
        AtomInfo = Spectra.Atom("Silicon",'SiII_1193','MaanIonisationTables/si2.hdf5',  1/T.SnapTimes[0]-1  )
        AtomInfo = Spectra.Atom("Silicon",'SiIV_1402','MaanIonisationTables/si4.hdf5',  1/T.SnapTimes[0]-1  )
        AtomInfo = Spectra.Atom("Silicon",'SiIII_1206','MaanIonisationTables/si3.hdf5',  1/T.SnapTimes[0]-1  )
        AtomInfo = Spectra.Atom("Iron",'FeII_1144','MaanIonisationTables/fe2.hdf5',  1/T.SnapTimes[0]-1  )
        AtomInfo = Spectra.Atom("Silicon",'SiII_1260','MaanIonisationTables/si2.hdf5',  1/T.SnapTimes[0]-1  )
        AtomInfo = Spectra.Atom("Oxygen",'OI_1302','MaanIonisationTables/o1.hdf5',  1/T.SnapTimes[0]-1  )
        AtomInfo = Spectra.Atom("Oxygen",'OVII_21','MaanIonisationTables/o7.hdf5',  1/T.SnapTimes[0]-1  )

        Atom refer to the below dicts IonIndex and IonMu. Linename refers to the keys in Misc/LineDict.hdf5, and 'MaanIonisationTables/h1.hdf5' a ionization table from Maan Hani
        """
        IonIndex = {}
        IonIndex["Hydrogen"]=0
        IonIndex["Helium"]=1
        IonIndex["Carbon"]=2
        IonIndex["Nitrogen"]=3
        IonIndex["Oxygen"]=4
        IonIndex["Neon"]=5
        IonIndex["Magnesium"]=6
        IonIndex["Silicon"]=7
        IonIndex["Iron"]=8

        IonMu = {}
        IonMu["Hydrogen"]=1.0
        IonMu["Helium"]=4.0
        IonMu["Carbon"]=12.0
        IonMu["Nitrogen"]=14.0
        IonMu["Oxygen"]=16.0
        IonMu["Neon"]=20.0
        IonMu["Magnesium"]=24.0
        IonMu["Silicon"]=28.0
        IonMu["Iron"]=56.0

        if '1.' in IonisationFilename:
            self.IonFlag = 1
        elif '2.' in IonisationFilename:
            self.IonFlag = 2
        elif '3.' in IonisationFilename:
            self.IonFlag = 3
        elif '4.' in IonisationFilename:
            self.IonFlag = 4
        elif '5.' in IonisationFilename:
            self.IonFlag = 5
        elif '6.' in IonisationFilename:
            self.IonFlag = 6
        elif '7.' in IonisationFilename:
            self.IonFlag = 7
        elif '8.' in IonisationFilename:
            self.IonFlag = 8
        else:
            print( 'We couldnt not set the IonFlag (which is used by DoIonisation). We better exit.')
            sys.exit()




        self.Atom = Atom
        self.Mu = IonMu[Atom]
        self.Linename = Linename
        self.IonisationFilename = IonisationFilename
        self.ArepoOutputIndex = IonIndex[Atom]
        self.Redshift = Redshift
        self.SnapTime = 1.0/(1.0+Redshift)
        self.IonTableClass = mhh_ion_tables.IonTableArepo(self.IonisationFilename)#Maans ionisation table
        self.IonTableClass.set_iz( Redshift )#set redshift

        LineDict,_=Papaya.ReadPapayaHDF5('Misc/LineDict.hdf5')
        self.Line_Wavelength,self.Line_f,self.Line_gamma = LineDict[Linename]

        self.IonFrac = numpy.array([None])

        self.h = 0.7#Hubble constant parameter

    def CalculateIonNumberDensity(self,GasDensity,GasMetals ):
        """
        Input:
            - GasDensity (in 1e10*Msun/ (kpc/h)**3), i.e. the Density-field in Arepo hdf5
            - GasMetals from Arepo hdf5
        Output:
            - Number density of ion in (1/cm**3).
        """

        #import IPython
        #IPython.embed()

        if self.IonFrac.shape[0] != GasDensity.shape[0]:
            print( 'Please run DoIonisation to set self.IonFrac before running CalculateIonNumberDensity!')
            print( 'Terminating!')
            sys.exit()
        GasDensity_PhysUnits = GasDensity.astype(numpy.float64)*1.0e10/self.h*const.M_sun/(self.SnapTime*u.kpc/self.h)**3
        GasDensityIon_PhysUnits = GasDensity_PhysUnits*self.IonFrac*GasMetals[:,self.ArepoOutputIndex]#Change index here!
        GasDensityIon_PhysUnits = GasDensityIon_PhysUnits.astype(numpy.float64)

        GasNumberDensityIon = (GasDensityIon_PhysUnits/self.Mu/const.m_p).to(1/u.cm**3).value
        return GasNumberDensityIon


    def CalculateIonMass(self,GasMass,GasMetals ):
        """
        Input:
            - GasMass, i.e. the Masses-field in Arepo hdf5
            - GasMetals from Arepo hdf5
        Output:
            - GasMass of ion in same units as Input.
        """

        #import IPython
        #IPython.embed()

        if self.IonFrac.shape[0] != GasMass.shape[0]:
            print( 'Please run DoIonisation to set self.IonFrac before running CalculateIonNumberDensity!')
            print( 'Terminating!')
            sys.exit()
        
        
        return (self.IonFrac*GasMass*GasMetals[:,self.ArepoOutputIndex]).astype(numpy.float64)


    
    
    
    
    
    def DoIonisation(self,log10GasNumberDensityH, log10GasTemp,log10GasMetallicity, JAGN, GasSFR):
        print( 'We are doing the ionisation modelling with DoIonisation(). We have set self.IonFlag to',self.IonFlag)
        self.IonFrac = self.IonTableClass.interp(log10GasNumberDensityH, log10GasTemp,log10GasMetallicity, JAGN, GasSFR, self.IonFlag)
        return self.IonFrac




def H(a,x):
    ""
    P  = x**2
    H0 = numpy.exp(-x**2)
    Q  = 1.5/x**2
    return H0 - a / numpy.sqrt(pi) / P * ( H0 * H0 * (4. * P * P + 7. * P + 4. + Q) - Q - 1.0 )

def LineFlux(Lambda, Lambda0, N, b,f,gam,z):
    c  = 2.998e10  #;cm/s
    m_e = 9.1095e-28# ;g
    e  = 4.8032e-10# ;cgs units

    C_a  = numpy.sqrt(pi) * e**2 * f * Lambda0 * 1.e-8 / m_e / c / b
    a = Lambda0 * 1.e-8 * gam / (4.*pi*b)

    dl_D = b/c*Lambda0
    x = (Lambda/(z+1.0)-Lambda0)/dl_D+0.01

    tau  = C_a*N*H(a,x)
    return numpy.exp(-tau)
