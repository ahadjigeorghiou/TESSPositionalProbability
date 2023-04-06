import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.mast import Catalogs
from astropy.io import fits
from . import TESSselfflatten as tsf
import numpy as np
import os
from pathlib import Path
from requests.exceptions import ConnectionError
import matplotlib.pyplot as plt
    

def load_default(infile, lc_dir, per_lim=None, depth_lim=None):
    if lc_dir == 'default':
        filepath = Path(__file__).resolve().parents[1] / 'Lightcurves' 
    else:
        filepath = Path(lc_dir)
    
    # Default data should be a csv file with ticid, toi/candidate num, per, t0, tdur, depth in this order
    data = pd.read_csv(f'Input/{infile}')
    data.columns = ['ticid', 'candidate', 'per', 't0', 'tdur', 'depth']
    data.set_index(['ticid', 'candidate'], inplace=True)
    data['lcdir'] = [filepath / str(ticid) for ticid in data.index.get_level_values('ticid')]
    
    data.loc[data['t0'] > 2457000, 't0'] -= 2457000
    
    sectors = []
    
    for ticid in data.index.unique('ticid'):
        files_loc = data.loc[ticid, 'lcdir'].iloc[0]
        files = list(files_loc.glob('*lc.fits*'))
        
        # Obtain the sector from the filename of each lc file
        sec = [int(f.stem.split('_')[4][-2:]) for f in files]
        # Order the sectors in ascending order
        sec = sorted(sec)
        
        sectors.append((ticid, sec))
        
    sec_df = pd.DataFrame(sectors, columns=['ticid', 'sectors']).set_index('ticid')
    
    data = data.join(sec_df, how='inner')
    data.sort_values(['ticid', 'candidate'])

    return data

def load_archive_toi(infile, lc_dir, per_lim=None, depth_lim=None):
    if lc_dir == 'default':
        filepath = Path(__file__).resolve().parents[1] / 'Lightcurves'   
    else:
        filepath = Path(lc_dir)
        
    cols = ['toi', 'tid', 'tfopwg_disp', 'pl_tranmid', 'pl_orbper', 'pl_trandurh', 'pl_trandep']
    
    toi_df = pd.read_csv(Path.cwd() / 'Input' / infile, usecols=cols)

    toi_df.columns = ['candidate', 'ticid', 'tfop_disp', 't0', 'per', 'tdur', 'depth']
    
    toi_df['candidate'] = [int(str(x).split('.')[1]) for x in toi_df['candidate']]
    
    toi_df['t0'] -= 2457000
    
    toi_df['tdur'] /= 24
    
    toi_df.loc[np.isnan(toi_df['per']), 'per'] = 0
    
    toi_df['lcdir'] = [filepath / str(ticid) for ticid in toi_df['ticid']]
    
    toi_df.set_index(['ticid', 'candidate'], inplace=True)
    
    sectors = []
    
    for ticid in toi_df.index.unique('ticid'):
        files_loc = toi_df.loc[ticid, 'lcdir'].iloc[0]
        if files_loc.exists():
            files = list(files_loc.glob('*lc.fits*'))
            
            # Obtain the sector from the filename of each lc file
            sec = [int(f.stem.split('_')[4][-2:]) for f in files]
            # Order the sectors in ascending order
            sec = sorted(sec)
        else:
            sec = np.nan
            
        sectors.append((ticid, sec))
        
    sec_df = pd.DataFrame(sectors, columns=['ticid', 'sectors']).set_index('ticid')
    
    toi_df = toi_df.join(sec_df, how='inner')
    toi_df.sort_values(['ticid', 'candidate'])
    
    toi_df = toi_df[~toi_df['sectors'].isnull()]
    
    if per_lim is not None:
        toi_df.query((f'per <= {per_lim}'), inplace=True)
    
    if depth_lim is not None:
        toi_df.query((f'depth >= {depth_lim}'), inplace=True)
        
    return toi_df


def load_exofop_toi(infile, lc_dir, per_lim=None, depth_lim=None):
    if lc_dir == 'default':
        filepath = Path(__file__).resolve().parents[1] / 'Lightcurves'   
    else:
        filepath = Path(lc_dir)
        
    cols = ['TIC ID', 'TOI', 'TESS Disposition', 'TFOPWG Disposition', 'Transit Epoch (BJD)', 'Period (days)', 'Duration (hours)', 'Depth (ppm)']
    
    toi_df = pd.read_csv(Path(__file__).resolve().parents[1] / 'Input' / infile, usecols=cols)
    
    toi_df.columns = ['ticid', 'candidate', 'tess_disp', 'tfop_disp', 't0', 'per', 'tdur', 'depth']
    
    toi_df['candidate'] = [int(str(x).split('.')[1]) for x in toi_df['candidate']]
    
    toi_df['t0'] -= 2457000
    
    toi_df['tdur'] /= 24
    
    toi_df['lcdir'] = [filepath / str(ticid) for ticid in toi_df['ticid']]
    
    toi_df.set_index(['ticid', 'candidate'], inplace=True)
    
    sectors = []
    
    for ticid in toi_df.index.unique('ticid'):
        files_loc = toi_df.loc[ticid, 'lcdir'].iloc[0]
        if files_loc.exists():
            files = list(files_loc.glob('*lc.fits*'))
            
            # Obtain the sector from the filename of each lc file
            sec = [int(f.stem.split('_')[4][-2:]) for f in files]
            # Order the sectors in ascending order
            sec = sorted(sec)
        else:
            sec = np.nan
            
        sectors.append((ticid, sec))
        
    sec_df = pd.DataFrame(sectors, columns=['ticid', 'sectors']).set_index('ticid')
    
    toi_df = toi_df.join(sec_df, how='inner')
    toi_df.sort_values(['ticid', 'candidate'])
    
    toi_df = toi_df[~toi_df['sectors'].isnull()]
    
    if per_lim is not None:
        toi_df.query((f'per <= {per_lim}'), inplace=True)
    
    if depth_lim is not None:
        toi_df.query((f'depth >= {depth_lim}'), inplace=True)
        
    return toi_df


def TIC_byID(ID):
    """
    """
    try:
        catTable = Catalogs.query_criteria(ID=ID, catalog="Tic")
        
        return catTable['ID', 'ra', 'dec', 'Tmag', 'e_Tmag', 'GAIAmag', 'e_GAIAmag', 'Vmag', 'e_Vmag']
    except ConnectionError:
        print('Connection failed. Source not created.')
        return 0


def TIC_lookup(coords, search_radius=0.05555):
    """
    """
    scoord = SkyCoord(ra=coords[0], dec=coords[1], unit='deg', frame='icrs')
    radius = u.Quantity(search_radius, u.deg)

    try:
        catTable = Catalogs.query_region(scoord, catalog="Tic", radius=radius) 
        
        return catTable['ID', 'ra', 'dec', 'rad', 'mass', 'Teff', 'Tmag', 'GAIAmag', 'Vmag', 'disposition']
    except ConnectionError:
        print('Connection error. Nearby sources not found.')
        return 0


def load_spoc_centroid(filepath, flatten=False, trim=False, cut_outliers=False, sectorstart=None, transitcut=False, tc_per=None, tc_t0=None, tc_tdur=None):
    """
    Loads TESS SPOC lightcurve centroid data
    """
    flag = ''
    
    hdu = fits.open(filepath)
    time = hdu[1].data['TIME']
    flux = hdu[1].data['PDCSAP_FLUX']
    if sectorstart is None:
        sectorstart = time[0]
    if tc_per == 0:
        tc_per = time[-1] - time[0]
    if tc_tdur / tc_per > 0.2:
        tc_tdur = tc_per*0.2
    X = hdu[1].data['MOM_CENTR1']
    Y = hdu[1].data['MOM_CENTR2']
    cam = hdu[0].header['CAMERA']
    ccd = hdu[0].header['CCD']   
    
    nancut = np.isnan(time) | np.isnan(X) | np.isnan(Y) | np.isnan(flux)
    time = time[~nancut]
    X = X[~nancut]
    Y = Y[~nancut]
    
    if len(time) <= len(hdu[1].data['TIME'])/2 or len(hdu[1].data['TIME']) < 3:  # if nancut removed > half the points, or there weren't any anyway
        flag = 'Not enough data'
    

    hdu.close()
    
    if flag:
        return time, X, Y, flag, cam, ccd
    
    normphase = np.abs((np.mod(time-tc_t0-tc_per*0.5, tc_per) - 0.5*tc_per) / (0.5*tc_tdur))
    intransit = normphase <= 1
    
    exp_time = np.nanmedian(np.diff(time)) * 24 * 60
    exp_time = int(np.round(exp_time))
    
    if sum(intransit) == 0 and not flag:
        flag = 'No transit data' 
    elif sum(intransit) <= 2 and not flag:
        flag = 'Not enough transit data'

    if trim:        
        if exp_time <= 2.1:
            cut_num = 360
        elif exp_time <= 11:
            cut_num = 72
        else:
            cut_num = 24
            
        # Remove the first 12 hours of observations
        time = time[cut_num:len(time)]
        X = X[cut_num:len(X)]
        Y = Y[cut_num:len(Y)]
        
        normphase = np.abs((np.mod(time-tc_t0-tc_per*0.5, tc_per) - 0.5*tc_per) / (0.5*tc_tdur))
        intransit = normphase <= 1
        
        if sum(intransit) == 0 and not flag:
            flag = 'No transit data after trim' 
        elif sum(intransit) <= 2 and not flag:
            flag = 'Not enough transit data after trim'
            
    if flatten:
        Xcurve = np.array([time, X, np.ones(len(time))]).T
        Ycurve = np.array([time, Y, np.ones(len(time))]).T

        X = tsf.TESSflatten(Xcurve, sectorstart=sectorstart, split=True, winsize=2, stepsize=0.15, polydeg=3,
                            niter=10, sigmaclip=4., gapthresh=100., transitcut=transitcut,
                            tc_per=tc_per, tc_t0=tc_t0, tc_tdur=tc_tdur, divide=False)
        Y = tsf.TESSflatten(Ycurve, sectorstart=sectorstart, split=True, winsize=2, stepsize=0.15, polydeg=3,
                            niter=10, sigmaclip=4., gapthresh=100., transitcut=transitcut,
                            tc_per=tc_per, tc_t0=tc_t0, tc_tdur=tc_tdur, divide=False)
        
    if cut_outliers:

        MAD_X = MAD(X[~intransit])
        MAD_Y = MAD(Y[~intransit])
        if (MAD_X == 0 or MAD_Y == 0) and not flag:
            flag = 'MAD is 0'

        cut = (np.abs(X-np.median(X))/MAD_X < cut_outliers) & (np.abs(Y-np.median(Y))/MAD_Y < cut_outliers)

        # avoid removing too much (happens with discontinuities for example)
        while np.sum(cut) < 3*len(X)/4:
            cut_outliers = cut_outliers + 1
            cut = (np.abs(X-np.median(X))/MAD_X < cut_outliers) & (np.abs(Y-np.median(Y))/MAD_Y < cut_outliers)
            
        cut[intransit] = True  # never cut points in the transit, too much risk they'll be marked as outliers
        time = time[cut]
        X = X[cut]
        Y = Y[cut]
        
        normphase = np.abs((np.mod(time-tc_t0-tc_per*0.5, tc_per) - 0.5*tc_per) / (0.5*tc_tdur))
        intransit = normphase <= 1
        
        mad_x_in = MAD(X[intransit])
        mad_y_in = MAD(Y[intransit])
        index_in = np.arange(len(time))[intransit]
        
        cut = (np.abs(X[intransit]-np.median(X[intransit]))/mad_x_in > 6) | (np.abs(Y[intransit]-np.median(Y[intransit]))/mad_y_in > 6)
        index_cut = index_in[cut]
        
        X = np.delete(X, index_cut) 
        Y = np.delete(Y, index_cut) 
        time = np.delete(time, index_cut) 
               
    return time, X, Y, flag, cam, ccd


def load_spoc_lc(filepath, hdu=None, flatten=False,  sectorstart=None, transitcut=False, tc_per=None, tc_t0=None, tc_tdur=None, return_trend=False, return_hdu=False):
    """
    Loads a TESS SPOC lightcurve, normalised with NaNs removed.
 
    Returns:
    lc -- 	dict
 		Lightcurve with keys time, flux, error. Error is populated with zeros.
    """
    if hdu is None:
        hdu = fits.open(filepath)
    time = hdu[1].data['TIME']

    if 'PDCSAP_FLUX' in hdu[1].columns.names:
        flux = hdu[1].data['PDCSAP_FLUX']
        err = hdu[1].data['PDCSAP_FLUX_ERR']
    else:
        flux = hdu[1].data['SAP_FLUX']
        err = np.zeros(len(flux))
    nancut = np.isnan(time) | np.isnan(flux) | (flux == 0)
    lc = {}
    lc['time'] = time[~nancut]
    lc['flux'] = flux[~nancut]
    lc['error'] = err[~nancut]
     
    norm = np.median(lc['flux'])
    lc['median'] = norm
           
    lcurve = np.array([lc['time'], lc['flux'], np.ones(len(lc['time']))]).T
    
    lcflat, trend = tsf.TESSflatten(lcurve, sectorstart=sectorstart, split=True, winsize=2.0, stepsize=0.15, polydeg=3,
                                niter=10, sigmaclip=4., gapthresh=100., transitcut=transitcut,
                                tc_per=tc_per, tc_t0=tc_t0, tc_tdur=tc_tdur, return_trend=True)
      
    
    if flatten:    
        lc['flux'] = lcflat
         
        mad = MAD(lc['flux'])
        
        if isinstance(tc_per, np.ndarray) or isinstance(tc_per, list):
            intransit = np.zeros(len(lc['time']), dtype=bool)
            for i, p in enumerate(tc_per):
                if p == 0:
                    p1 = lc['time'][-1] - (tc_t0[i]-tc_tdur[i]*0.5)
                    p2 = tc_t0[i] - (lc['time'][0]-tc_tdur[i]*0.5)
                    p = np.max((p1, p2))
                normphase = np.abs((np.mod(lc['time']-tc_t0[i]-p*0.5, p) - 0.5*p) / (0.5*tc_tdur[i]))
                intransit += normphase <= 1
        else:
            normphase = np.abs((np.mod(lc['time']-tc_t0-tc_per*0.5, tc_per) - 0.5*tc_per) / (0.5*tc_tdur))
            intransit = normphase <= 1
        
        # Remove intransit upward outliers
        outliers = (lc['flux'] - np.median(lc['flux']))/mad > 4
        # outliers[~intransit] = False
                        
        lc['time'] = lc['time'][~outliers]
        lc['flux'] = lc['flux'][~outliers]
        lc['error'] = lc['error'][~outliers]
        
    else:
        lc['flux'] = lc['flux']/norm
    
    trend = trend/norm
    lc['error'] = lc['error']/norm

    if return_hdu and return_trend:
        return lc, trend, hdu
    elif return_hdu and not return_trend:
        return lc, hdu
    elif not return_hdu and return_trend:
        hdu.close()
        del hdu
        
        return lc, trend
    else:
        hdu.close()
        del hdu
        
        return lc
         
         
def load_spoc_masks(filepath):
    hdu = fits.open(filepath)
    wcs = WCS(hdu[2].header)
    mask_data = hdu[2].data
    cam = hdu[0].header['CAMERA']
    ccd = hdu[0].header['CCD']
    
    origin = hdu[2].header['CRVAL1P'], hdu[2].header['CRVAL2P']
    
    # Use the bit information to retrieve the aperture or centroid masks. 
    # The centroid mask will be the same as the aperture for the majority of cases.
    aperture = np.bitwise_and(mask_data, 2) / 2
    centroid = np.bitwise_and(mask_data, 8) / 8
    
    hdu.close()
    del hdu
    return aperture, centroid, wcs, origin, cam, ccd


def phasefold(time, per, t0=0):
    return np.mod(time - t0, per) / per


def MAD(array):
    """
    Median Average Deviation
    """
    mednorm = np.nanmedian(array)
    return 1.4826 * np.nanmedian(np.abs(array - mednorm))


def find_index(array, value):
    return np.argmin(np.abs(array - value))


def observed_transits(time, t0, per, tdur):
    '''Returns the number of observed transits within a time window'''

    # Find the transit epoch occuring right before the observation window
    while t0 > time[0]:
        t0 -= per

    # Find the first transit occuring after the start of the observation window
    while t0 < time[0]:
        t0 += per

    # Now check the number of observed transits within the observation window
    mid_point = t0
    count = 0
    while mid_point < time[-1]:
        # Find the index of the element in the time array closest to the mid transit point
        t_idx = find_index(time, mid_point)

        # Check if the difference between the time at the closest index and the mid transit point is less than half a transit duration
        if abs(time[t_idx] - mid_point) < tdur*0.5:
            # If yes count the transit as observed
            count += 1

        # Advance to the next transit
        mid_point += per

    return count


def centroid_fitting(ticid, candidate, sector, time, X, Y, per, t0, tdur, tdur23, loss='linear', plot=False): 
    if per == 0:
        per = time[-1] - time[0]
        
    normphase = np.abs((np.mod(time-t0-per*0.5, per) - 0.5*per) / (0.5*tdur))
    
    intransit = normphase <= 1  
    intransit_half = normphase <= 0.5 # avoids half of transit to minimise ingress affecting result
    nearby = (normphase > 1) & (normphase < 3)
    
    if sum(intransit_half) == 0:
        flag = 'No half transit points'
    elif sum(intransit_half) < 3:
        flag = 'Not enough half transit points'
    elif sum(nearby) < 6:
        flag = 'No nearby points'
    else:
        flag = ''
    
    if not flag:            
        from scipy import optimize
        def _Trapezoidmodel(phase_data, t23, t14, depth):
            t0_phase = 0.5
            centrediffs = np.abs(phase_data - t0_phase)
            model = np.zeros_like(phase_data)
            model[centrediffs<t23/2.] = depth
            in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
            model[in_gress] = depth + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*(-depth)
            if t23>t14:
                model = np.ones(len(phase_data))*1e8
            return model   

        phase = phasefold(time,per, t0+per*0.5)  #transit at phase 0.5
        idx = np.argsort(phase)
        phase = phase[idx]
               
        initialguess = [tdur23 / per, tdur / per, 0]
        bounds=[(initialguess[0]*0.95, initialguess[1]*0.99, -np.inf),(initialguess[0]*1.05, initialguess[1]*1.01, np.inf)]
        
        try:
            xfit = optimize.curve_fit(_Trapezoidmodel, phase, X[idx], 
                                    p0=initialguess, sigma=np.full_like(X, MAD(X[~intransit])),
                                    bounds=bounds,
                                    absolute_sigma=False, loss=loss)
                            
            yfit = optimize.curve_fit(_Trapezoidmodel, phase, Y[idx], 
                                    p0=initialguess, sigma=np.full_like(Y, MAD(Y[~intransit])),
                                    bounds=bounds,
                                    absolute_sigma=False, loss=loss)       
        
            x_diff = xfit[0][2]
            x_err = np.sqrt(np.diag(xfit[1]))[2] 
            y_diff = yfit[0][2]
            y_err = np.sqrt(np.diag(yfit[1]))[2]
            
            if x_err >= 1 or y_err >= 1:
                x_diff, x_err, y_diff, y_err = np.nan, np.nan, np.nan, np.nan
                flag = 'Fit did not converge'
            
            if plot:
                modelX = _Trapezoidmodel(phase, *xfit[0])
                modelY = _Trapezoidmodel(phase, *yfit[0])

                phase -= 0.5
                
                limit = np.abs(phase) < tdur*4/per
                
                fig, ax = plt.subplots(1,1, figsize=(8,8))
                fig.suptitle(f'TIC {ticid} - {candidate} Sector {sector}')
                ax.scatter(phase[limit], X[idx][limit], s=0.5, c='k')
                ax.plot(phase[limit], modelX[limit], c='darkorange', lw=2)
                ax.scatter(phase[limit], Y[idx][limit] - 0.01, s=0.5, c='k')
                ax.plot(phase[limit], modelY[limit]-0.01, c='darkorange', lw=2)
                ax.set_xlabel('Phase', fontsize=14)
                ax.set_ylabel('Normalized Centroid Position', fontsize=14)
                
                outfile = Path(__file__).resolve().parents[1] / 'Output' / 'Plots' / f'{ticid}' 
                outfile.mkdir(exist_ok=True)
                outfile = outfile / f'centroidfit_{ticid}_{candidate}_{sector}.png'
                fig.savefig(outfile, bbox_inches='tight')
                 
        except Exception:
            x_diff, x_err, y_diff, y_err = np.nan, np.nan, np.nan, np.nan
            flag = 'Fit fail'
    else:
        x_diff, x_err, y_diff, y_err = np.nan, np.nan, np.nan, np.nan
    
    return x_diff, x_err, y_diff, y_err, flag


def nearby_depth(depth, f_t, f_n):
    depth_n = depth * np.divide(f_t, f_n, out=np.zeros_like(f_n), where=f_n!=0.0)

    return depth_n


def test_target_aperture(x, y, aperture):
    xint = int(np.floor(x+0.5))
    yint = int(np.floor(y+0.5))
        
    if xint < 0 or xint >= aperture.shape[1] or yint < 0 or yint >= aperture.shape[0]:
        test = False
    else:
        test = bool(aperture[yint, xint])
        
    return test
    

def calc_flux_fractions(sec, cam, ccd, origin, X_sources, Y_sources, fluxes, aperture):
    from CandidateSet import PRF
    
    # Identify the pixels used in the aperture
    pixels = np.where(aperture == 1)

    # Extract a 2x2 array of the mask grid. Pixels not used will retain 0 in their values.
    y_min = np.min(pixels[0])
    y_max = np.max(pixels[0]) + 1
    
    x_min = np.min(pixels[1])
    x_max = np.max(pixels[1]) + 1
    
    aperture_only = aperture[y_min:y_max, x_min:x_max]
    
    aperture_flux = np.zeros([X_sources.size, aperture_only.shape[0], aperture_only.shape[1]])
        
    tp_x = np.round(origin[0] + X_sources[0])
    tp_y = np.round(origin[1] + Y_sources[0])

    if sec < 4:
        prf_folder = Path(__file__).resolve().parents[0] / 'PRF files' / 'Sector 1'
    else:
        prf_folder = Path(__file__).resolve().parents[0] / 'PRF files' / 'Sector 4'
                
    prf = PRF.TESS_PRF(cam, ccd, sec, tp_x, tp_y, prf_folder)
    
    for i in range(len(X_sources)):
        aperture_flux[i] = prf.locate(X_sources[i], Y_sources[i], aperture.shape)[y_min:y_max, x_min:x_max]
        
    aperture_flux *= fluxes[:, None, None] #convert fractions to actual fluxes within each pixel (still by star)    
    
    aperture_flux *= aperture_only # Multiplies by either 1 or 0 if pixel is in the aperture mask or not
    all_ap_flux = np.sum(aperture_flux)
    
    flux_fractions = []
    
    for i in range(len(X_sources)):
        source_ap_flux = np.sum(aperture_flux[i])
        flux_fractions.append(source_ap_flux/all_ap_flux)
        
    return flux_fractions, all_ap_flux  


def prf_fractions(sec, cam, ccd, origin, X_sources, Y_sources, aperture):
    from CandidateSet import PRF
    
    # Identify the pixels used in the aperture
    pixels = np.where(aperture == 1)

    # Extract a 2x2 array of the mask grid. Pixels not used will retain 0 in their values.
    y_min = np.min(pixels[0])
    y_max = np.max(pixels[0]) + 1
    
    x_min = np.min(pixels[1])
    x_max = np.max(pixels[1]) + 1
    
    aperture_only = aperture[y_min:y_max, x_min:x_max]
    fluxfractions = np.zeros([X_sources.size, aperture_only.shape[0], aperture_only.shape[1]])
    
    tp_x = np.round(origin[0] + X_sources[0])
    tp_y = np.round(origin[1] + Y_sources[0])

    if sec < 4:
        prf_folder = Path(__file__).resolve().parents[0] / 'PRF files' / 'Sector 1'
    else:
        prf_folder = Path(__file__).resolve().parents[0] / 'PRF files' / 'Sector 4'
                
    prf = PRF.TESS_PRF(cam, ccd, sec, tp_x, tp_y, prf_folder)
    
    for i in range(len(X_sources)):
        fluxfractions[i] = prf.locate(X_sources[i], Y_sources[i], aperture.shape)[y_min:y_max, x_min:x_max]
        
    return fluxfractions


def model_centroid(aperture, fluxes, fluxfractions):
    # Identify the pixels used in the aperture
    pixels = np.where(aperture == 1)

    # Extract a 2x2 array of the mask grid. Pixels not used will retain 0 in their values.
    y_min = np.min(pixels[0])
    y_max = np.max(pixels[0]) + 1
    
    x_min = np.min(pixels[1])
    x_max = np.max(pixels[1]) + 1
    
    aperture_only = aperture[y_min:y_max, x_min:x_max]
    fluxfractions = fluxfractions * fluxes[:, None, None] #convert fractions to actual fluxes within each pixel (still by star)
    fluxfractions = np.sum(fluxfractions, axis=0) #should now be 2D. All flux contributions in each pixel now totalled. Could be used for diagnostics, this array should now be a simulated image of the aperture.
 
    fluxfractions *= aperture_only # Multiplies by either 1 or 0 if pixel is in the aperture mask or not
    #need to specify indices in below two lines. Depends on format of aperture.
    X = np.average(np.arange(x_min, x_max),weights=np.sum(fluxfractions,axis=0)) #sums fluxfractions across y axis to leave x behind
    Y = np.average(np.arange(y_min, y_max),weights=np.sum(fluxfractions,axis=1)) #sums fluxfractions across x axis to leave y behind
    
    return X, Y    


def calc_centroid_probability(cent_x, cent_y, cent_x_err, cent_y_err, diff_x, diff_y, diff_x_err, diff_y_err):
    X_err = np.sqrt(cent_x_err ** 2 + diff_x_err ** 2)
    Y_err = np.sqrt(cent_y_err ** 2 + diff_y_err ** 2)
    
    from scipy import spatial
    # get mahalanobis distance
    cov = np.array([[X_err ** 2, 0], [0, Y_err ** 2]])
    VI = np.linalg.inv(cov)
    mahalanobis = spatial.distance.mahalanobis([diff_x, diff_y], [cent_x, cent_y], VI)

    prob_centroid = np.exp(-(mahalanobis ** 2) / 2)
    
    return prob_centroid
