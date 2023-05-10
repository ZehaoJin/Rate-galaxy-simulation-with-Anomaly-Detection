import pynbody
import glob,os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from tqdm import tqdm
import scipy
from astropy.visualization import make_lupton_rgb
from matplotlib import colors

## using Keri's cmd file


#def homemade_calc_mags(simstars, band='r', cmd_path='/home/zj448/Radiative_transfer/SDSS_cmd.npz'):
def homemade_calc_mags(simstars, band='r', cmd_path='bpass_ugriz_sin_imf135_300.npz'):

    # load a table of magnitute[age,metalicity]
    lums=np.load(cmd_path)

    # get age and metal in this simulation
    age_star = simstars['age'].in_units('yr')
    metals = simstars['metals']

    # clip age and metal that gets out of table's range
    age_star[age_star < np.min(lums['ages'])] = np.min(lums['ages'])
    age_star[age_star > np.max(lums['ages'])] = np.max(lums['ages'])
    metals[metals < np.min(lums['mets'])] = np.min(lums['mets'])
    metals[metals > np.max(lums['mets'])] = np.max(lums['mets'])

    # make the table into grid for interpolation
    age_grid = np.log10(lums['ages'])
    met_grid = lums['mets']
    mag_grid = lums[band]

    # 2d interpolate. note that age is converted to "dex" (10^dex)
    # my cmd
    #output_mags = scipy.interpolate.interpn((age_grid,met_grid),mag_grid,(np.log10(age_star),metals))
    # Keri's cmd
    output_mags = scipy.interpolate.interpn((met_grid,age_grid),mag_grid,(metals,np.log10(age_star)))

    # http://stev.oapd.inaf.it/cgi-bin/cmd
    # http://stev.oapd.inaf.it/cmd_3.6/help.html
    # the magnitudes we get here is "Single-burst stellar populations (SSP), integrated magnitudes (for 1 Msun)"
    # "In CMD, the integrated magnitudes of single-burst stellar population (SSP) are computed for a unit mass of the stellar population
    #  initially born, for the given IMF. They are computed for all filters in the selected photometric system. SP integrated magnitudes
    #  are derived assuming stars populate continuously the entire isochrone,and hence do not include the stochastic variations in the
    #  integrated magnitudes (and colours) that are typical of real SSPs (star clusters)."

    # I guess the following is trying to get the magnitude for the actual mass of our star, instead of 1 Msun
    # https://en.wikipedia.org/wiki/Absolute_magnitude --> Bolometric magnitude. here vals=Mag_star, output_mags=Mag_sun
    # https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation   mass-luminosity relation
    # seems like here it takes alpha=1

    try:
        vals = output_mags - 2.5 * np.log10(simstars['massform'].in_units('Msol'))
    except KeyError as ValueError:
        vals = output_mags - 2.5 * np.log10(simstars['mass'].in_units('Msol'))
    # make it a SimArray
    vals=pynbody.array.SimArray(vals)
    vals.units = None
    return vals



#def homemade_lum_den(simstars,band='r',cmd_path='/home/zj448/Radiative_transfer/SDSS_cmd.npz'):
def homemade_lum_den(simstars,band='r',cmd_path='bpass_ugriz_sin_imf135_300.npz'):
    mag=homemade_calc_mags(simstars,band=band,cmd_path=cmd_path)
    val = (10 ** (-0.4 * mag)) * simstars['rho'] / simstars['mass']
    val.units = simstars['rho'].units/simstars['mass'].units
    return val

def homemade_clipping(r,g,b,min_perc):
    maxrgb=np.array([r.max(),g.max(),b.max()]).max()
    if min_perc is None:
        return r/maxrgb,g/maxrgb,b/maxrgb
    else:
        minium=maxrgb*min_perc
        r-=minium
        g-=minium
        b-=minium
        r[r<0]=0
        g[g<0]=0
        b[b<0]=0
        maxrgb=np.array([r.max(),g.max(),b.max()]).max()
        return r/maxrgb,g/maxrgb,b/maxrgb

#@pynbody.derived_array
#def SDSS_bol(sim):
    #return homemade_calc_mags(sim,band='bol')
@pynbody.derived_array
def SDSS_u(sim):
    return homemade_calc_mags(sim,band='u')
@pynbody.derived_array
def SDSS_g(sim):
    return homemade_calc_mags(sim,band='g')
@pynbody.derived_array
def SDSS_r(sim):
    return homemade_calc_mags(sim,band='r')
@pynbody.derived_array
def SDSS_i(sim):
    return homemade_calc_mags(sim,band='i')
@pynbody.derived_array
def SDSS_z(sim):
    return homemade_calc_mags(sim,band='z')

#@pynbody.derived_array
#def SDSS_bol_lum_den(sim):
    #return homemade_lum_den(sim,band='bol')
@pynbody.derived_array
def SDSS_u_lum_den(sim):
    return homemade_lum_den(sim,band='u')
@pynbody.derived_array
def SDSS_g_lum_den(sim):
    return homemade_lum_den(sim,band='g')
@pynbody.derived_array
def SDSS_r_lum_den(sim):
    return homemade_lum_den(sim,band='r')
@pynbody.derived_array
def SDSS_i_lum_den(sim):
    return homemade_lum_den(sim,band='i')
@pynbody.derived_array
def SDSS_z_lum_den(sim):
    return homemade_lum_den(sim,band='z')

'''
def homemade_render(sim,
                    r_band='SDSS_i', g_band='SDSS_r', b_band='SDSS_g',
                    rgb_scale=[1,1,1],
                    min_intensity_frac=None,
                    width='10 kpc',
                    resolution=500,
                    starsize=None,
                    plot=True, axes=None, ret_im=False,
                    with_dust=True,z_range=50.0):
'''
def homemade_render(sim,
                    r_band='SDSS_i', g_band='SDSS_r', b_band='SDSS_g',
                    width='10 kpc',
                    resolution=500,
                    starsize=None,
                    with_dust=True,z_range=50.0):
    # handle all kinds of input for width. e.g. '15 kpc'
    if isinstance(width, str) or issubclass(width.__class__, pynbody.units.UnitBase):
        if isinstance(width, str):
            width = pynbody.units.Unit(width)
        width = width.in_units(sim['pos'].units, **sim.conversion_context())


    if starsize is not None:
        smf = pynbody.filt.HighPass('smooth', str(starsize) + ' kpc')
        sim.s[smf]['smooth'] = pynbody.array.SimArray(starsize, 'kpc', sim=sim)


    r = pynbody.plot.sph.image(sim.s, qty=r_band + '_lum_den', width=width, log=False,
                               units="pc^-2", clear=False, noplot=True, resolution=resolution)
    g = pynbody.plot.sph.image(sim.s, qty=g_band + '_lum_den', width=width, log=False,
                               units="pc^-2", clear=False, noplot=True, resolution=resolution)
    b = pynbody.plot.sph.image(sim.s, qty=b_band + '_lum_den', width=width, log=False,
                               units="pc^-2", clear=False, noplot=True, resolution=resolution)


    #'''
    ##########  mag/parsec^2 to mag/arcsec^2
    #  he is assuming some distance here?
    r=pynbody.plot.stars.convert_to_mag_arcsec2(r)
    g=pynbody.plot.stars.convert_to_mag_arcsec2(g)
    b=pynbody.plot.stars.convert_to_mag_arcsec2(b)


    # render image with a simple dust absorption correction based on Calzetti's law using the gas content.
    if with_dust is True:
        try:
            import extinction
        except ImportError:
            warnings.warn(
                "Could not load extinction package. If you want to use this feature, "
                "plaese install the extinction package from here: http://extinction.readthedocs.io/en/latest/"
                "or <via pip install extinction> or <conda install -c conda-forge extinction>", RuntimeWarning)
            return

        warm = pynbody.filt.HighPass('temp',3e4)
        cool = pynbody.filt.LowPass('temp',3e4)
        positive = pynbody.filt.BandPass('z',-z_range,z_range) #LowPass('z',0)

        column_den_warm = pynbody.plot.sph.image(sim.g[positive][warm], qty='rho', width=width, log=False,
                                                 units="kg cm^-2", clear=False, noplot=True,z_camera=z_range)
        column_den_cool = pynbody.plot.sph.image(sim.g[positive][cool], qty='rho', width=width, log=False,
                                                 units="kg cm^-2", clear=False, noplot=True,z_camera=z_range)
        mh = 1.67e-27 # units kg

        cool_fac = 0.25 # fudge factor to make dust absorption not too strong
        # get the column density of gas
        col_den = np.divide(column_den_warm,mh)+np.divide(column_den_cool*cool_fac,mh)
        # get absorption coefficient
        a_v = 0.5*col_den*2e-21

        #get the central wavelength for the given band
        wavelength_avail = {'SDSS_u':3561.79,'SDSS_g':4718.87,'SDSS_r':6185.19,'SDSS_i':7499.70,'SDSS_z':8961.49} #in Angstrom

        lr,lg,lb = wavelength_avail[r_band],wavelength_avail[g_band],wavelength_avail[b_band] #in Angstrom
        wave = np.array([lb, lg, lr])

        ext_r = np.empty_like(r)
        ext_g = np.empty_like(g)
        ext_b = np.empty_like(b)

        for i in range(len(a_v)):
            for j in range(len(a_v[0])):
                ext = extinction.calzetti00(wave.astype(np.float64), a_v[i][j].astype(np.float64), 3.1, unit='aa', out=None)
                ext_r[i][j] = ext[2]
                ext_g[i][j] = ext[1]
                ext_b[i][j] = ext[0]

        r = r+ext_r
        g = g+ext_g
        b = b+ext_b

    # magnitude/arcsec^2 to intensity or should I call it lum_den?
    r=10**(-r/2.5)
    g=10**(-g/2.5)
    b=10**(-b/2.5)

    pc2_to_sqarcsec = 2.3504430539466191e-09
    r=r/pc2_to_sqarcsec
    g=g/pc2_to_sqarcsec
    b=b/pc2_to_sqarcsec
    #'''



    '''
    # scaling
    # https://cosmo.nyu.edu/hogg/visualization/rgb/nw_scale_rgb.pro
    r,g,b=pynbody.plot.stars.nw_scale_rgb(r,g,b,rgb_scale)

    # arcsinh
    # https://arxiv.org/pdf/astro-ph/0312483.pdf
    # https://cosmo.nyu.edu/hogg/visualization/rgb/nw_arcsinh_fit.pro
    r,g,b=pynbody.plot.stars.nw_arcsinh_fit(r,g,b)

    # normalize to range 0 to 1
    r,g,b=homemade_clipping(r,g,b,min_perc=min_intensity_frac)

    # make it an rgb image array
    rgbim=np.dstack((r,g,b))

    if plot is True:
        if axes is None:
            axes = plt.gca()
        if axes:
            axes.imshow(rgbim[::-1, :], extent=(-width / 2, width / 2, -width / 2, width / 2))
            axes.set_xlabel('x [' + str(sim.s['x'].units) + ']')
            axes.set_ylabel('y [' + str(sim.s['y'].units) + ']')
            plt.draw()

    if ret_im:
        return rgbim
    '''

    return r,g,b
    #rgbim=make_lupton_rgb(r, g, b)

    #if plot is True:
        #plt.imshow(rgbim, origin='lower')
    #if ret_im:
        #return rgbim

#useage
# rgbim=homemade_render(h[1].s,width='10 kpc',with_dust=True,ret_im=True,plot=True,rgb_scale=[6.5,5.25,3.])
def homemade_normalize(r,g,b):
    maxrgb=np.array([r.max(),g.max(),b.max()]).max()
    minrgb=np.array([r.min(),g.min(),b.min()]).min()
    r=(r-minrgb)/(maxrgb-minrgb)
    g=(g-minrgb)/(maxrgb-minrgb)
    b=(b-minrgb)/(maxrgb-minrgb)
    return r,g,b
