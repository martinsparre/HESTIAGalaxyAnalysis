import Papaya,numpy,sys

Dict = {}

F = open('atompar_ale.dat','r+')


for line in F:
    tmp = line.split()

    Dict[tmp[1]] = numpy.array([float(tmp[0]),float(tmp[2]), float(tmp[3])])#Wavelength [Angstrom], f-parameter, gamma-parameter


Header={}
Header['Info'] = """
code for printing wavelength, f and gamma for SiII 1526:
import Papaya
Dict,_=Papaya.ReadPapayaHDF5('LineDict.hdf5')
print Dict['SiII_1526']
"""
Header['Code'] = open(sys.argv[0], 'r').read()
Header['atompar_ale.dat'] = open('atompar_ale.dat', 'r').read()


Papaya.WritePapayaHDF5('LineDict.hdf5',Dict,Header)
