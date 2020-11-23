"""
Arepo ion tables

A module to handle the FG UVB & AGN ionization field to calculate the
ionization balance using MHH photoionization equilibrium lookup tables.


MHH: Redshift bug is fixed and interpolating over metallicity array!!

"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#from yt.utilities.on_demand_imports import _h5py as h5py
import h5py
import numpy as np




def h5rd( fname, path, dtype=None ):
    """ Read Data. Return a dataset located at <path> in file <fname> as
    a numpy array.
    e.g. rd( fname, '/PartType0/Coordinates' ). """

    data = None
    with h5py.File( fname, 'r' ) as h5f:
        ds = h5f[path]
        if dtype is None:
            dtype = ds.dtype
        #data = np.zeros( ds.shape, dtype=dtype )
        #data = ds.value
        data = ds[()]
    return data



class IonTableArepo:

    """ A class to handle Arepo ionization tables. """

    # MHH: defining them automatically: see __init__
    # MHH: need to change Delta_nH and delta_T?? add Delta_Jagn
    #DELTA_nH   = 0.25
    #DELTA_T    = 0.1
    #DELTA_Jagn = 0.25

    def __init__(self, ion_file):

        self.ion_file = ion_file

        # ionbal is indexed like [nH, T, z]
        # nH and T are log quantities
        #---------------------------------------------------------------
        self.nH   = h5rd( ion_file, '/logd' )          # log nH [cm^-3]
        self.T    = h5rd( ion_file, '/logt' )          # log T [K]
        self.z    = h5rd( ion_file, '/redshift' )      # z
        # MHH:  (log!!!)   set zero to -40 (it's skipped anyway)
        self.Jagn = h5rd( ion_file, '/logJagn')[1:]    # log AGN intensity [erg s^-1 cm^-2]
        # MHH: adding metallicity
        self.met  = h5rd( ion_file, '/logZ')

        # read the ionization fractions
        # linear values stored in file so take log here
        # ionbal is the ionization balance (i.e. fraction)
        #---------------------------------------------------------------
        self.ionbal       = h5rd( ion_file, '/ionbal' ).astype(np.float64)
        # MHH: ignoring metallcity for now (logZ = -1.0)
        ###self.ionbal_AGN   = self.ionbal.copy()[:,:,:,1:]
        ###self.ionbal_noAGN = self.ionbal.copy()[:,:,:,0]
        self.ionbal_AGN   = self.ionbal.copy()[:,:,:,:,1:]
        self.ionbal_noAGN = self.ionbal.copy()[:,:,:,:,0]

        # MHH: defining delta's
        self.DELTA_nH   = np.diff(self.nH)[-1]
        self.DELTA_T    = np.diff(self.T)[-1]
        self.DELTA_met  = np.diff(self.met)[-1]
        self.DELTA_Jagn = np.diff(self.Jagn)[-1]
        self.DELTA_z    = np.diff(self.z)[-1]




        # MHH: our ionbal is in log so no need to take log10
        '''
        ipositive = np.where( self.ionbal > 0.0 )
        izero = np.where( self.ionbal <= 0.0 )
        self.ionbal[izero] = self.ionbal[ipositive].min()

        self.ionbal = np.log10( self.ionbal )
        '''

        # MHH: doesn't seem to be used
        '''
        # calculate the spacing along each dimension
        #---------------------------------------------------------------
        self.dnH   = self.nH[1:]   - self.nH[0:-1]
        self.dT    = self.T[1:]    - self.T[0:-1]
        self.dz    = self.z[1:]    - self.z[0:-1]
        # MHH: calculate spacing in AGN intensities
        self.dJagn = self.Jagn[1:] - self.Jagn[0:-1]


        # MHH: need to figure out order of arrays
        self.order_str = '[log nH, log T, z, log Jagn]'
        '''




    # sets iz and fz
    #-----------------------------------------------------
    def set_iz( self, z ):
        if z <= self.z[0]:
            self.iz = 0
            self.fz = 0.0
        elif z >= self.z[-1]:
            self.iz = len(self.z) - 2
            self.fz = 1.0
        else:
            for iz in range( len(self.z)-1 ):
                if z < self.z[iz+1]:
                    self.iz = iz
                    self.fz = ( z - self.z[iz] ) / self.DELTA_z
                    break


    # interpolate the table at a fixed redshift for the input
    # values of nH and T ( input should be log ).  A simple
    # tri-linear interpolation is used. Z_met_in is in units of solar metallicity.
    #-----------------------------------------------------
    def interp( self, nH_in, T_in, Zmet_in, Jagn_in, SFR_in, ion_state_in):
        nH = np.array( nH_in )
        T  = np.array( T_in )
        # MHH: adding Jagn array  NOT LOG!  -- linear gets log-ed later
        Jagn = np.array(Jagn_in)
        met = np.array(Zmet_in)

        if nH.size != T.size != Jagn.size != met.size:
            raise ValueError(' mhh_ion_tables: array size mismatch !!! ')

        # field discovery will have nH.size == 1 , T.size == 1, Jagn.size == 1
        # in that case we simply return 1.0
        if nH.size == 1 and T.size == 1 and Jagn.size == 1 and met.size == 1:
            ionfrac = 1.0
            return ionfrac

        # MHH: setting up for ionfrac calculations
        ionfrac = np.zeros_like(nH)

        # MHH: creating a mask for particles with Jagn == 0
        AGN_mask   = np.argwhere(Jagn >  0.)[:,0]
        noAGN_mask = np.argwhere(Jagn == 0.)[:,0]

        #
        ##
        ### calculate ionfrac for particles with Jagn > 0.
        ##
        #

        nH   = np.array(nH_in[AGN_mask])
        T    = np.array(T_in[AGN_mask])
        Jagn = np.array(np.log10(Jagn_in[AGN_mask]))
        met  = np.array(Zmet_in[AGN_mask])


        # find inH and fnH
        #-----------------------------------------------------
        x_nH = ( nH - self.nH[0] ) / self.DELTA_nH
        x_nH_clip = np.clip( x_nH, 0.0, self.nH.size-1.001 )
        fnH,inH = np.modf( x_nH_clip )
        inH = inH.astype( np.int32 )


        # find iT and fT
        #-----------------------------------------------------
        x_T = ( T - self.T[0] ) / self.DELTA_T
        x_T_clip = np.clip( x_T, 0.0, self.T.size-1.001 )
        fT,iT = np.modf( x_T_clip )
        iT = iT.astype( np.int32 )


        # find iJagn and fJagn
        #-----------------------------------------------------
        x_Jagn = ( Jagn - self.Jagn[0] ) / self.DELTA_Jagn
        x_Jagn_clip = np.clip( x_Jagn, 0.0, self.Jagn.size-1.001 )
        fJagn,iJagn = np.modf( x_Jagn_clip )
        iJagn = iJagn.astype( np.int32 )

        # find imet and fmet
        #-----------------------------------------------------
        x_met = ( met - self.met[0] ) / self.DELTA_met
        x_met_clip = np.clip( x_met, 0.0, self.met.size-1.001 )
        fmet,imet = np.modf( x_met_clip )
        imet = imet.astype( np.int32 )



        # short names for previously calculated iz and fz
        #-----------------------------------------------------
        iz = self.iz
        fz = self.fz


        # calculate interpolated value
        # use tri-linear interpolation on the log values
        #-----------------------------------------------------

        # including metallicity interpolation
        ionfrac[AGN_mask] = self.ionbal_AGN[inH  , iT  , imet  , iz  , iJagn  ] * (1-fnH) * (1-fT) * (1-fmet) * (1-fz) * (1-fJagn) +\
                            self.ionbal_AGN[inH+1, iT  , imet  , iz  , iJagn  ] * ( fnH ) * (1-fT) * (1-fmet) * (1-fz) * (1-fJagn) +\
                            self.ionbal_AGN[inH  , iT+1, imet  , iz  , iJagn  ] * (1-fnH) * ( fT ) * (1-fmet) * (1-fz) * (1-fJagn) +\
                            self.ionbal_AGN[inH  , iT  , imet+1, iz  , iJagn  ] * (1-fnH) * (1-fT) * ( fmet ) * (1-fz) * (1-fJagn) +\
                            self.ionbal_AGN[inH  , iT  , imet  , iz+1, iJagn  ] * (1-fnH) * (1-fT) * (1-fmet) * ( fz ) * (1-fJagn) +\
                            self.ionbal_AGN[inH  , iT  , imet  , iz  , iJagn+1] * (1-fnH) * (1-fT) * (1-fmet) * (1-fz) * ( fJagn ) +\
                            self.ionbal_AGN[inH+1, iT+1, imet  , iz  , iJagn  ] * ( fnH ) * ( fT ) * (1-fmet) * (1-fz) * (1-fJagn) +\
                            self.ionbal_AGN[inH+1, iT  , imet+1, iz  , iJagn  ] * ( fnH ) * (1-fT) * ( fmet ) * (1-fz) * (1-fJagn) +\
                            self.ionbal_AGN[inH+1, iT  , imet  , iz+1, iJagn  ] * ( fnH ) * (1-fT) * (1-fmet) * ( fz ) * (1-fJagn) +\
                            self.ionbal_AGN[inH+1, iT  , imet  , iz  , iJagn+1] * ( fnH ) * (1-fT) * (1-fmet) * (1-fz) * ( fJagn ) +\
                            self.ionbal_AGN[inH  , iT+1, imet+1, iz  , iJagn  ] * (1-fnH) * ( fT ) * ( fmet ) * (1-fz) * (1-fJagn) +\
                            self.ionbal_AGN[inH  , iT+1, imet  , iz+1, iJagn  ] * (1-fnH) * ( fT ) * (1-fmet) * ( fz ) * (1-fJagn) +\
                            self.ionbal_AGN[inH  , iT+1, imet  , iz  , iJagn+1] * (1-fnH) * ( fT ) * (1-fmet) * (1-fz) * ( fJagn ) +\
                            self.ionbal_AGN[inH  , iT  , imet+1, iz+1, iJagn  ] * (1-fnH) * (1-fT) * ( fmet ) * ( fz ) * (1-fJagn) +\
                            self.ionbal_AGN[inH  , iT  , imet+1, iz  , iJagn+1] * (1-fnH) * (1-fT) * ( fmet ) * (1-fz) * ( fJagn ) +\
                            self.ionbal_AGN[inH  , iT  , imet  , iz+1, iJagn+1] * (1-fnH) * (1-fT) * (1-fmet) * ( fz ) * ( fJagn ) +\
                            self.ionbal_AGN[inH+1, iT+1, imet+1, iz  , iJagn  ] * ( fnH ) * ( fT ) * ( fmet ) * (1-fz) * (1-fJagn) +\
                            self.ionbal_AGN[inH+1, iT+1, imet  , iz+1, iJagn  ] * ( fnH ) * ( fT ) * (1-fmet) * ( fz ) * (1-fJagn) +\
                            self.ionbal_AGN[inH+1, iT+1, imet  , iz  , iJagn+1] * ( fnH ) * ( fT ) * (1-fmet) * (1-fz) * ( fJagn ) +\
                            self.ionbal_AGN[inH+1, iT  , imet+1, iz+1, iJagn  ] * ( fnH ) * (1-fT) * ( fmet ) * ( fz ) * (1-fJagn) +\
                            self.ionbal_AGN[inH+1, iT  , imet+1, iz  , iJagn+1] * ( fnH ) * (1-fT) * ( fmet ) * (1-fz) * ( fJagn ) +\
                            self.ionbal_AGN[inH+1, iT  , imet  , iz+1, iJagn+1] * ( fnH ) * (1-fT) * (1-fmet) * ( fz ) * ( fJagn ) +\
                            self.ionbal_AGN[inH  , iT+1, imet+1, iz+1, iJagn  ] * (1-fnH) * ( fT ) * ( fmet ) * ( fz ) * (1-fJagn) +\
                            self.ionbal_AGN[inH  , iT+1, imet+1, iz  , iJagn+1] * (1-fnH) * ( fT ) * ( fmet ) * (1-fz) * ( fJagn ) +\
                            self.ionbal_AGN[inH  , iT+1, imet  , iz+1, iJagn+1] * (1-fnH) * ( fT ) * (1-fmet) * ( fz ) * ( fJagn ) +\
                            self.ionbal_AGN[inH  , iT  , imet+1, iz+1, iJagn+1] * (1-fnH) * (1-fT) * ( fmet ) * ( fz ) * ( fJagn ) +\
                            self.ionbal_AGN[inH+1, iT+1, imet+1, iz+1, iJagn  ] * ( fnH ) * ( fT ) * ( fmet ) * ( fz ) * (1-fJagn) +\
                            self.ionbal_AGN[inH+1, iT+1, imet+1, iz  , iJagn+1] * ( fnH ) * ( fT ) * ( fmet ) * (1-fz) * ( fJagn ) +\
                            self.ionbal_AGN[inH+1, iT+1, imet  , iz+1, iJagn+1] * ( fnH ) * ( fT ) * (1-fmet) * ( fz ) * ( fJagn ) +\
                            self.ionbal_AGN[inH+1, iT  , imet+1, iz+1, iJagn+1] * ( fnH ) * (1-fT) * ( fmet ) * ( fz ) * ( fJagn ) +\
                            self.ionbal_AGN[inH  , iT+1, imet+1, iz+1, iJagn+1] * (1-fnH) * ( fT ) * ( fmet ) * ( fz ) * ( fJagn ) +\
                            self.ionbal_AGN[inH+1, iT+1, imet+1, iz+1, iJagn+1] * ( fnH ) * ( fT ) * ( fmet ) * ( fz ) * ( fJagn )




        #
        ##
        ### calculate ionfrac for particles with no AGN radiation
        ##
        #
        nH   = np.array( nH_in[noAGN_mask] )
        T    = np.array( T_in[noAGN_mask] )
        met  = np.array( Zmet_in[noAGN_mask])

        # find inH and fnH
        #-----------------------------------------------------
        x_nH = ( nH - self.nH[0] ) / self.DELTA_nH
        x_nH_clip = np.clip( x_nH, 0.0, self.nH.size-1.001 )
        fnH,inH = np.modf( x_nH_clip )
        inH = inH.astype( np.int32 )


        # find iT and fT
        #-----------------------------------------------------
        x_T = ( T - self.T[0] ) / self.DELTA_T
        x_T_clip = np.clip( x_T, 0.0, self.T.size-1.001 )
        fT,iT = np.modf( x_T_clip )
        iT = iT.astype( np.int32 )


        # find imet and fmet
        #-----------------------------------------------------
        x_met = ( met - self.met[0] ) / self.DELTA_met
        x_met_clip = np.clip( x_met, 0.0, self.met.size-1.001 )
        fmet,imet = np.modf( x_met_clip )
        imet = imet.astype( np.int32 )


        # short names for previously calculated iz and fz
        #-----------------------------------------------------
        iz = self.iz
        fz = self.fz


        # calculate interpolated value
        # use tri-linear interpolation on the log values
        #-----------------------------------------------------
        #import IPython
        #IPython.embed()
        ionfrac[noAGN_mask] = self.ionbal_noAGN[inH  , iT  , imet  , iz  ] * (1-fnH) * (1-fT) * (1-fz) * (1-fmet) +\
                              self.ionbal_noAGN[inH  , iT+1, imet  , iz  ] * (1-fnH) * ( fT ) * (1-fz) * (1-fmet) +\
                              self.ionbal_noAGN[inH  , iT  , imet  , iz+1] * (1-fnH) * (1-fT) * ( fz ) * (1-fmet) +\
                              self.ionbal_noAGN[inH  , iT  , imet+1, iz  ] * (1-fnH) * (1-fT) * (1-fz) * ( fmet ) +\
                              self.ionbal_noAGN[inH+1, iT  , imet  , iz  ] * ( fnH ) * (1-fT) * (1-fz) * (1-fmet) +\
                              self.ionbal_noAGN[inH+1, iT+1, imet  , iz  ] * ( fnH ) * ( fT ) * (1-fz) * (1-fmet) +\
                              self.ionbal_noAGN[inH+1, iT  , imet  , iz+1] * ( fnH ) * (1-fT) * ( fz ) * (1-fmet) +\
                              self.ionbal_noAGN[inH+1, iT  , imet+1, iz  ] * ( fnH ) * (1-fT) * (1-fz) * ( fmet ) +\
                              self.ionbal_noAGN[inH  , iT+1, imet  , iz+1] * (1-fnH) * ( fT ) * ( fz ) * (1-fmet) +\
                              self.ionbal_noAGN[inH  , iT+1, imet+1, iz  ] * (1-fnH) * ( fT ) * (1-fz) * ( fmet ) +\
                              self.ionbal_noAGN[inH  , iT  , imet+1, iz+1] * (1-fnH) * (1-fT) * ( fz ) * ( fmet ) +\
                              self.ionbal_noAGN[inH  , iT+1, imet+1, iz+1] * (1-fnH) * ( fT ) * ( fz ) * ( fmet ) +\
                              self.ionbal_noAGN[inH+1, iT  , imet+1, iz+1] * ( fnH ) * (1-fT) * ( fz ) * ( fmet ) +\
                              self.ionbal_noAGN[inH+1, iT+1, imet+1, iz  ] * ( fnH ) * ( fT ) * (1-fz) * ( fmet ) +\
                              self.ionbal_noAGN[inH+1, iT+1, imet  , iz+1] * ( fnH ) * ( fT ) * ( fz ) * (1-fmet) +\
                              self.ionbal_noAGN[inH+1, iT+1, imet+1, iz+1] * ( fnH ) * ( fT ) * ( fz ) * ( fmet )

        #
        # MHH: SFR mask -- all SF gas is fully neutral
        #-----------------------------------------------------
        ionfrac = 10.**ionfrac

        SFR_mask = np.argwhere(SFR_in > 0.)[:,0]
        if int(ion_state_in) == 1:
            ionfrac[SFR_mask] = 1.
        elif int(ion_state_in) > 1:
            ionfrac[SFR_mask] = 0.

        return ionfrac
