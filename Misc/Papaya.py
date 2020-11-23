#Written by Martin Sparre, June 2016.

import h5py

def WritePapayaHDF5(Filename,Dict,Attrs={}):
    'Dict is a dictionary containing numpy-arrays. Attrs an optional Dict with e.g. header information.'

    F = h5py.File(Filename,'w')

    for key in Dict.keys():
        dset = F.create_dataset(key, Dict[key].shape, dtype=Dict[key].dtype)
        dset[:] = Dict[key]

    dset = F.create_dataset('Attrs', (0,), dtype=None)
    for key in Attrs.keys():
        dset.attrs[key] = Attrs[key]

    F.close()

def ReadPapayaHDF5(Filename,ReadDict = True):
    'input Filename previosly saved file. Outputs Dict and Attrs. Dict_ReadOnlyKey is a list of Dict keys to be read - if None, everything is read.'

    F = h5py.File(Filename,'r')

    Dict = {}
    for key in F.keys():
        if key != 'Attrs' and ReadDict == True:
            Dict[key] = F[key][()]

    Attrs = {}
    for key in F['Attrs'].attrs.keys():
        Attrs[key] = F['Attrs'].attrs.get(key)

    F.close()
    return Dict,Attrs

if __name__ == '__main__':
    import numpy
    A = {}
    A['Masses'] = numpy.zeros(1000)+0.3
    A['Pos'] = numpy.zeros((1000,3))
    A['Pos'][:,0] = 1.0
    A['Pos'][:,1] = 2.0
    A['Pos'][:,2] = 3.0
    A['ID'] = numpy.array(numpy.zeros((1000,3))+927,dtype=numpy.int64)

    Attrs = {}
    Attrs['Header'] = 'This is a simple test'
    Attrs['Snapshotnumber'] = 135
    Attrs['MassArray'] = numpy.array([1,2.0,3,4,5])

    WritePapayaHDF5('Test.hdf5',A,Attrs=Attrs)

    B,C = ReadPapayaHDF5('Test.hdf5')
