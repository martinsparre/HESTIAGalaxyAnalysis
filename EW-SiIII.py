#/usr/bin/python
#-*- coding: utf-8 -*-
import sys,time
import matplotlib
#matplotlib.use('Agg') #Enable this when running from Cluster account. This disables matplotlib.show() command.
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
SnapTime = 1.0#SnapTime is the scale factor
Redshift = 0.0
mu = (2*0.7381+0.75*0.2485+0.0134/2)**(-1)#mean molecular weight


###############################################################
#####  Initialise Atom class used for ion calculations ########
###############################################################

#Maan Hani has to be offerend coauthorship, if you use these functions. From https://ui.adsabs.harvard.edu/abs/2019MNRAS.488..135H/abstract and https://ui.adsabs.harvard.edu/abs/2018MNRAS.475.1160H/abstract

#Comment in different lines to do different ions. Here we do Ly alpha / HI.

#ThisAtom = Spectra.Atom("Hydrogen",'Ly_a','MaanIonisationTables/Martin200328_h1.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Oxygen",'OVI_1031','MaanIonisationTables/Martin200328_o6.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Oxygen",'OVII_21','MaanIonisationTables/Martin200328_o7.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Oxygen",'OVIII_2','MaanIonisationTables/Martin200328_o8.hdf5',1/T.SnapTimes[0]-1)#Note this absorption line does not exist, so spectrum should not be analysed. Nlos is still correct!
#ThisAtom = Spectra.Atom("Oxygen",'OI_1302','MaanIonisationTables/Martin200328_o1.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Carbon",'CIV_1548','MaanIonisationTables/Martin200328_c4.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Carbon",'CIV_1550','MaanIonisationTables/Martin200328_c4.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Silicon",'SiII_1193','MaanIonisationTables/Martin200328_si2.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Silicon",'SiII_1260','MaanIonisationTables/Martin200328_si2.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Silicon",'SiII_1526','MaanIonisationTables/Martin200328_si2.hdf5',1/T.SnapTimes[0]-1)
ThisAtom = Spectra.Atom("Silicon",'SiIII_1206','MaanIonisationTables/Martin200328_si3.hdf5',Redshift)
#ThisAtom = Spectra.Atom("Silicon",'SiIV_1402','MaanIonisationTables/Martin200328_si4.hdf5',1/T.SnapTimes[0]-1)
#ThisAtom = Spectra.Atom("Iron",'FeII_1144','MaanIonisationTables/Martin200328_fe2.hdf5',1/T.SnapTimes[0]-1)

###############################################################
############### a plot to play around with ####################
###############################################################


LambdaArray = numpy.linspace(ThisAtom.Line_Wavelength-2,ThisAtom.Line_Wavelength+2,1001)
N = 1.0e13#cm**-2
b = 10.0*1e5#cm/s
FluxArray = Spectra.LineFlux(LambdaArray, ThisAtom.Line_Wavelength, N, b,ThisAtom.Line_f,ThisAtom.Line_gamma,0.0)

EW = (FluxArray.size*1.0-numpy.sum(FluxArray) ) * (LambdaArray[1]-LambdaArray[0])

plt.figure(1)
plt.plot(LambdaArray,FluxArray,'-',lw=2,color='blue')
plt.fill_between(LambdaArray,FluxArray,y2=1.0,color='red')
plt.text(ThisAtom.Line_Wavelength+0.5,0.5,'EW=%4.4f Å'%EW)
plt.ylim((0.0,1.01))
plt.xlabel('Wavelength [Å]')
plt.ylabel('Normalized flux')
plt.savefig('EW_SiIII1206.pdf')
plt.show()

if FluxArray[0]<0.99:
    print('Warning, increase wavelength interval!!!!!')
    print()

###############################################################
#################### curve of growth (CoG) ####################
###############################################################



LambdaArray = numpy.linspace(ThisAtom.Line_Wavelength-20,ThisAtom.Line_Wavelength+20    ,40001,dtype=numpy.float128)

plt.figure(2)
#first 1 km/s
b = 1.0*1e5#cm/s
List_N = 10.0**numpy.linspace(6,18,81)
List_EW = []


for N in List_N:

    FluxArray = Spectra.LineFlux(LambdaArray, ThisAtom.Line_Wavelength, N, b,ThisAtom.Line_f,ThisAtom.Line_gamma,0.0)
    EW = (FluxArray.size*1.0-numpy.sum(FluxArray) ) * (LambdaArray[1]-LambdaArray[0])
    List_EW.append(EW)

plt.plot(numpy.log10(List_N),numpy.log10(List_EW),'-',lw=2,color='black',label='1 km s$^{-1}$')



b = 10.0*1e5#cm/s
List_EW = []
for N in List_N:
    FluxArray = Spectra.LineFlux(LambdaArray, ThisAtom.Line_Wavelength, N, b,ThisAtom.Line_f,ThisAtom.Line_gamma,0.0)
    EW = (FluxArray.size*1.0-numpy.sum(FluxArray) ) * (LambdaArray[1]-LambdaArray[0])
    List_EW.append(EW)

plt.plot(numpy.log10(List_N),numpy.log10(List_EW),'-',lw=2,color='blue',label='10 km s$^{-1}$')


b = 20.0*1e5#cm/s
List_EW = []
for N in List_N:
    FluxArray = Spectra.LineFlux(LambdaArray, ThisAtom.Line_Wavelength, N, b,ThisAtom.Line_f,ThisAtom.Line_gamma,0.0)
    EW = (FluxArray.size*1.0-numpy.sum(FluxArray) ) * (LambdaArray[1]-LambdaArray[0])
    List_EW.append(EW)

plt.plot(numpy.log10(List_N),numpy.log10(List_EW),'-',lw=2,color='green',label='20 km s$^{-1}$')

#then 60 km/s
b = 30.0*1e5#cm/s
List_EW = []
for N in List_N:
    FluxArray = Spectra.LineFlux(LambdaArray, ThisAtom.Line_Wavelength, N, b,ThisAtom.Line_f,ThisAtom.Line_gamma,0.0)
    EW = (FluxArray.size*1.0-numpy.sum(FluxArray) ) * (LambdaArray[1]-LambdaArray[0])
    List_EW.append(EW)

plt.plot(numpy.log10(List_N),numpy.log10(List_EW),'-',lw=2,color='orange',label='30 km s$^{-1}$')

b = 100.0*1e5#cm/s
List_EW = []
for N in List_N:
    FluxArray = Spectra.LineFlux(LambdaArray, ThisAtom.Line_Wavelength, N, b,ThisAtom.Line_f,ThisAtom.Line_gamma,0.0)
    EW = (FluxArray.size*1.0-numpy.sum(FluxArray) ) * (LambdaArray[1]-LambdaArray[0])
    List_EW.append(EW)

plt.plot(numpy.log10(List_N),numpy.log10(List_EW),'-',lw=2,color='red',label='100 km s$^{-1}$')
plt.legend(loc=4)
plt.grid()
plt.xlabel(r'$\log N_{\rm Si III}$ (cm$^{-2}$)')
plt.ylabel('log EW Si III 1206 (Å)')
plt.savefig('CurveOfGrowth_SiIII1206.pdf')
plt.show()
