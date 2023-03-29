import numpy as np
import pandas as pd
from pathlib import Path
import os

import astropy.units as u

from collections import defaultdict
import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import chain

from . import utils, TransitFit


class CandidateSet(object):

    def __init__(self, infile, infile_type='default', env='default', per_lim=None, depth_lim=None, multiprocessing=1, save_output=False, save_suffix=None, load_suffix=None):
        """
        Load in a set of candidates with their transit parameters [period, epoch, depth]. Sets up the environment for running the positional probabilitiy generation           
        
        Parameters
        infile - path to input file 
        infile_type - options: default/archive/exofop, allows for loading in data from specific databases or the default loading format
        env
        per_lim - set maximum period limit for candidates
        depth_lim - set minimum candidate 
        multiprocessing - set maximum number of workers (set 1 for no multiprocessing)
        save_output - True/False, affects all data generation
        save_suffix - Suffix for the filenames of all saved data
        load_suffix - Suffix for loading previously saved data 

        """

        if infile_type == 'exofop':
            self.data = utils.load_exofop_toi(infile, env, per_lim=per_lim, depth_lim=depth_lim)
        elif infile_type == 'archive':
            self.data = utils.load_archive_toi(infile, env, per_lim=per_lim, depth_lim=depth_lim)
        elif infile_type == 'default':
            self.data = utils.load_csetdata(infile, ftype=infile_type, env=env)
        else:
            raise ValueError('Infile type must be set as one of: default/archive/exofop')
        
        # Data containers    
        self.sources = {} # Source objects
        self.sector_data = None # Per sector transit data
        self.centroid = None # Observed centroid offsets
        self.probabilities = pd.DataFrame() # Positional Probabilities

        self.find_stars = False
        self.update_data = False
        self.centroid_data = False 
        self.flux_fractions = False
        self.estimate_depths = False
        self.possible_sources = False
        
        self.multiprocessing = multiprocessing
        
        self.save_output = save_output
        
        if save_output:   
            if save_suffix:
                self.save_suffix = save_suffix
            else:
                self.save_suffix = datetime.today().strftime('%d%m%Y_%H%M%S')
        
        self.load_suffix = load_suffix
            
    
    def find_TIC_stars(self, load_suffix_ovr=None, save_overwrite=None, rerun=False):
        """
        Identify nearby TIC stars (includes GAIA) for each candidate, up to 8.5 ΔTmag.
        
        Parameters:
        
        """
        # Get the unique ticids. Multiple candidates on one source will result to only one source created.
        targets = self.data.index.unique('ticid')

        # Load previous data if provided
        preload = False
        if self.load_suffix or load_suffix_ovr or infile:
            if load_suffix_ovr:
                infile = Path.cwd() / 'Output' / f'sources_{load_suffix_ovr}.pkl'
            else:
                infile = Path.cwd() / 'Output' / f'sources_{self.load_suffix}.pkl'
            print('Loading from ' + str(infile))
            try:
                with open(infile, 'rb') as f:
                    self.sources = pickle.load(f)

                # Compare loaded sources ids with the ids of the targets in the data to find any that might be missing
                targets = np.setdiff1d(targets, list(self.sources.keys()), assume_unique=True)
                preload = True
            except Exception as e:
                print(e)
                print('Error loading infile sources, recreating...')
                pass
        
        if not preload and not rerun:
            targets = np.setdiff1d(targets, list(self.sources.keys()), assume_unique=True)
            print('Existing sources found and are being reused')
            
        def _find(targetid):
            source_obj = None
            nearbyfail = 0
            s_rad = 8 * 21 * u.arcsec.to('degree')  # ~8 TESS pixels
            source_data = utils.TIC_byID(targetid)

            if not source_data:
                return source_obj, nearbyfail
                
            mags = {'TESS': (source_data['Tmag'][0], source_data['e_Tmag'][0]),
                    'G': (source_data['GAIAmag'][0], source_data['e_GAIAmag'][0]),
                    'V': (source_data['Vmag'][0], source_data['e_Vmag'][0])}
                
            source_obj = Source(tic = targetid,
                                coords = (source_data['ra'][0], source_data['dec'][0]), 
                                mags = mags)

            ticsources = utils.TIC_lookup(source_obj.coords, search_radius=s_rad)
            
            if not ticsources:
                nearbyfail = 1
                
            nearsources = defaultdict(list)
            for source_data in ticsources:
                if source_data['disposition'] == 'ARTIFACT' or source_data['disposition'] == 'DUPLICATE':
                    if int(source_data['ID']) == targetid:
                        return None, source_data['disposition']
                    else:
                        continue
                    
                nearsources['ticid'].append(int(source_data['ID']))
                nearsources['ra'].append(source_data['ra'])
                nearsources['dec'].append(source_data['dec'])
                nearsources['rad'].append(source_data['rad'])
                nearsources['mass'].append(source_data['mass'])
                nearsources['teff'].append(source_data['Teff'])
                nearsources['Tmag'].append(source_data['Tmag'])
                nearsources['G'].append(source_data['GAIAmag'])
                nearsources['V'].append(source_data['Vmag'])
            
            # Convert dictionary to dataframe for ease of usage
            nearsources = pd.DataFrame(nearsources)
            
            # Remove faint sources with dTmag greater than 8.5
            tmag_limit = source_obj.mags['TESS'][0] + 8.5
            nearsources.query(f'Tmag < {tmag_limit}', inplace=True)
            nearsources.set_index('ticid', inplace=True)
            
            nearsources['Flux'] = 15000 * 10 ** (-0.4 * (nearsources['Tmag'] - 10))
            
            # Ensure that the target is always first in the dataframe
            nearsources['order'] = np.arange(len(nearsources))  + 1
            nearsources.loc[targetid, 'order'] = 0
            nearsources.sort_values('order', inplace=True)
            nearsources.drop('order', axis=1, inplace=True)
            source_obj.nearby_data = nearsources
            
            return source_obj, nearbyfail
        
        
        if len(targets) > 0:
            print(f'Retrieving data from MAST for {len(targets)} targets')
            source_fail = []
            nearby_fail = []
            duplicate = []
            artifact = []
            if self.multiprocessing > 1 and len(targets) > 30:
                workers = np.min((self.multiprocessing*4, 20))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    try:
                        futures = {ex.submit(_find, targetid): targetid for targetid in targets}
                        
                        for future in as_completed(futures):
                            ticid = futures[future]
                            try:
                                source_obj, near = future.result()
                                if source_obj:
                                    self.sources[ticid] = source_obj
                                else:
                                    if near == 'DUPLICATE':
                                        print(f'{targetid} skipped due to being flagged as a du')
                                        duplicate.append(targetid)
                                    elif near == 'ARTIFACT':
                                        print(f'{targetid} skipped due to being flagged as a du')
                                        artifact.append(targetid)
                                    else:
                                        source_fail.append(targetid)
                                if near == 1:
                                    nearby_fail.append(targetid)
                            except Exception as e:
                                source_fail.append(ticid)
                                nearby_fail.append(ticid)
           
                    except KeyboardInterrupt:
                        ex.shutdown(wait=False, cancel_futures=True)
                        # ex._threads.clear()
                        # thread._threads_queues.clear()
                        raise ValueError('Keyboard interrupt')
                    
            else:
                for targetid in targets:
                    source_obj, near = _find(targetid)
                    if source_obj:
                        self.sources[targetid] = source_obj
                    else:
                        if near == 'Duplicate':
                            print(f'{targetid} skipped due to being a Duplicate')
                            duplicate.append(targetid)
                        else:
                            source_fail.append(targetid)
                    if near == 1:
                        nearby_fail.append(targetid)
            
            # Remove duplicate sources
            self.data.drop(duplicate, axis=0, level=0, inplace=True)
            # Output completion log
            print('Source identification and object creation completed.')
            if len(duplicate) > 0:
                print(f'{len(duplicate)} duplicate source(s) removed:', duplicate)
            if len(artifact) > 0:
                print(f'{len(artifact)} artifact source(s) removed:', artifact)
                
            print(f'{len(source_fail)} source(s) failed to be created:', source_fail)
            print(f'Failure to find nearby stars for {len(nearby_fail)} source(s)', nearby_fail)

        # Save output to be reused
        if self.save_output:
            outfile = Path.cwd() / 'Output' / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)

        self.find_stars = True
        
    
    def generate_per_sector_data(self, load_suffix_ovr=None, save_overwrite=None, rerun=False):
        if not self.find_stars:
            raise ValueError('Run find_tic_stars first')

        preload = False
        if load_suffix_ovr or self.load_suffix:
            if load_suffix_ovr:
                infile = Path.cwd() / 'Output' / f'sectordata_{load_suffix_ovr}.csv'
            else:
                infile = Path.cwd() / 'Output' / f'sectordata_{self.load_suffix}.csv'
            print('Loading from ' + str(infile))
            try:
                data_preloaded = pd.read_csv(infile).set_index(['ticid', 'candidate', 'sector'])
                preload = True
            except:
                print('Error loading infile sector data, recreating..')
        
        # Create empty multi-index
        indx = pd.MultiIndex(names=['ticid', 'candidate', 'sector'], levels=[[],[],[]], codes=[[],[],[]])
        for ticid in self.sources.keys():
            cndts = self.data.loc[ticid].index
            sectors = self.data.loc[ticid].sectors.iloc[0]
            indx = indx.append(pd.MultiIndex.from_product([[ticid], cndts, sectors], names=['ticid','candidate','sector']))
        
        # Create empty dataframe with the multi-index constructed above
        new_df = pd.DataFrame(data=0, index=indx, columns=['t0', 'per', 'sec_tdur', 'tsec_dur23', 'sec_depth'])
        
        if not rerun:
            if not preload:
                # Check if sector_data already exists and update the newly constructed dataframe with existing values
                if self.sector_data is not None:
                    # Either as a dataframe for all sources
                    new_df = pd.concat([new_df, self.sector_data])
                    new_df = new_df[~new_df.index.duplicated(keep='last')]   
            else:           
                # Update the dataframe with the preloaded values
                new_df = pd.concat([new_df, data_preloaded])
                new_df = new_df[~new_df.index.duplicated(keep='last')]
        
        self.sector_data = new_df
        # Find the entries which are still zero
        to_fill = self.sector_data.query('init_t0 == 0')
        ticids = to_fill.index.unique('ticid')
        num_targets = len(ticids)
        
        if num_targets > 0:
            print(f'Running sector_data update for {num_targets} targets')
            if self.multiprocessing > 1 and num_targets > 5:
                print('Runing multiprocessing')
                if num_targets < self.multiprocessing:
                    workers = num_targets
                else:
                    workers = self.multiprocessing
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    try:
                        factor = 20
                        while num_targets < 5*factor*workers:
                            factor -= 1
                            if factor == 1:
                                break
                        ticid_split = np.array_split(ticids, factor*workers)
                        print(len(ticid_split))  
                        futures = {ex.submit(self.update_multi_target, ticid_group): ticid_group for ticid_group in ticid_split}
                        
                        for future in as_completed(futures):
                            try:
                                filled, fails = future.result()
                                self.sector_data = pd.concat([self.sector_data, filled])
                                self.sector_data = self.sector_data[~self.sector_data.index.duplicated(keep='last')]
                                if len(fails) > 0:
                                    for fail in fails:
                                        print(f'Exception {fail[1]} occur with ticid: {fail[0]}')
                            except Exception as e:
                                group = futures[future]
                                print(f'Exception {e} occur with ticid group: {group}')
                    except KeyboardInterrupt:
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise ValueError('Keyboard interrupt')            
            else:
                ticid_split = np.array_split(ticids, int(np.ceil(len(ticids)/10)))
                for ticid_group in ticid_split:
                    filled, fail = self.update_multi_target(ticid_group)
                    
                    self.sector_data = pd.concat([self.sector_data, filled])
                    self.sector_data = self.sector_data[~self.sector_data.index.duplicated(keep='last')]
                
                    for ticid, e in fail:
                        print(f'Exception {e} occur with ticid: {ticid}')
                                                   
            if self.save_output:
                if save_overwrite:
                    outfile = Path.cwd() / 'Output' / f'sectordata_{save_overwrite}.csv'
                else:
                    outfile = Path.cwd() / 'Output' / f'sectordata_{self.save_suffix}.csv'
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                self.sector_data.to_csv(outfile)
                
        self.update_data = True
        
        
    def per_sector_data(self, ticid, initial_data):
        # Calculate per candidate/sector depth, tdur, tdur23
        
        lcdir = initial_data.iloc[0].lcdir
        sectors = initial_data.iloc[0].sectors
        candidates = initial_data.index.to_numpy()
        
        lcs = {}
        for sec in sectors:
            lcfile = list(lcdir.glob(f'*{sec:04}*lc.fits'))[0]
            lc = utils.TESSload(lcfile, flatten=True, sectorstart=None, transitcut=True,
                                    tc_per=initial_data.per.values, tc_t0=initial_data.t0.values,
                                    tc_tdur=initial_data.tdur.values)
            lc['error'] = np.full_like(lc['flux'], utils.MAD(lc['flux']))
            lcs[sec] = lc

        
        sector_data = []
        
        for cndt in candidates:
            cols = ['ticid', 'candidate', 'sector', 't0','per', 'sec_tdur', 'sec_tdur23', 'sec_depth']
            df = pd.DataFrame(index=range(len(sectors)), columns=cols)
            # Retrieve provided t0, per, tdur, depth per candidate
            cndt_t0 = initial_data.loc[cndt, 't0']
            cndt_per = initial_data.loc[cndt, 'per']
            cndt_tdur = initial_data.loc[cndt, 'tdur']
            cndt_depth = initial_data.loc[cndt, 'depth']
            
            df['ticid'] = ticid
            df['candidate'] = cndt
            df['sector'] = sectors
            df['t0'] = cndt_t0
            df['per'] = cndt_per
            
            if cndt_per == 0 or np.isnan(cndt_per):
                if (cndt_t0 > lc['time'][-1]) or (cndt_t0 < lc['time'][0]):

                    df['sec_tdur'] = cndt_tdur
                    df['tdur23'] = np.nan
                    df['sec_depth'] = np.nan

                    sector_data.append(df)
             
                    continue
                else:
                    cndt_per1 = lc['time'][-1] - (cndt_t0-cndt_tdur*0.5)
                    cndt_per2 = cndt_t0 - (lc['time'][0]-cndt_tdur*0.5)
                    cndt_per = np.max((cndt_per1, cndt_per2))
            
            sec_tdur = []
            sec_tdur23 = []
            sec_depth = []        
            for sec in sectors:
                sec_lc = lcs[sec]
                transits = utils.observed_transits(sec_lc['time'], cndt_t0, cndt_per, cndt_tdur)
                phase = utils.phasefold(sec_lc['time'], cndt_per, cndt_t0 - 0.5*cndt_per) -0.5
                intransit = np.abs(phase) < 0.5*cndt_tdur/cndt_per
                
                if transits > 0 and sum(intransit) > 3:
                    trapfit_initialguess = np.array([cndt_t0, cndt_tdur * 0.9 / cndt_per, cndt_tdur / cndt_per, cndt_depth*1e-6])
                    exp_time = np.median(np.diff(sec_lc['time']))
                    trapfit = TransitFit.TransitFit(sec_lc, trapfit_initialguess, exp_time, sfactor=7, fittype='trap', fixper=cndt_per, fixt0=cndt_t0)
                    
                    sec_tdur.append(trapfit.params[1]*cndt_per)
                    sec_tdur23.append(trapfit.params[0]*cndt_per)
                    sec_depth.append(trapfit.params[2]*1e6)
                else:
                    sec_tdur.append(cndt_tdur)
                    sec_tdur23.append(np.nan)
                    sec_depth.append(np.nan)

            df['sec_tdur'] = sec_tdur
            df['tdur23'] = sec_tdur23
            df['sec_depth'] = sec_depth
            
            sector_data.append(df)
                                                     
        return sector_data
    
    
    def multi_sector_data(self, ticids):
        results = []
        fails = []
        for ticid in ticids:
            try:
                results.append(self.update_target_transit_data(ticid, self.data.loc[ticid]))
            except Exception as e:
                fails.append((ticid, e))
        
                                    
        results = list(chain.from_iterable(results))
        results.append(pd.DataFrame(columns=['ticid', 'candidate', 'sector', 't0', 'per', 'sec_tdur', 'tsec_dur23', 'sec_depth']))
        
        filled = pd.concat(results, ignore_index=True)
        filled.set_index(['ticid', 'candidate', 'sector'], inplace=True)
        
        return filled, fails
    
    def generate_centroiddata(self, load_suffix=None, infile=None, save_overwrite=None, rerun=False):
        """
        Calculate the centroid difference in eclipse for this candidate set
        """
        if not self.update_data:
            raise ValueError('Run find_tic_stars first')
        
        preload = False
        if load_suffix or infile:
            if infile:
                infile = Path(infile)
            else:
                infile = Path.cwd() / 'Output' / f'centroiddata_{load_suffix}.csv'
            print('Loading from ' + str(infile))
            try:
                centroid_preloaded = pd.read_csv(infile).set_index(['ticid', 'candidate', 'sector'])
                centroid_preloaded.loc[centroid_preloaded['flag'].isna(), 'flag'] = ''
                preload = True
            except:
                print('Error loading infile centroid data, recreating..')
        
        # Create empty multi-index
        indx = pd.MultiIndex(names=['ticid', 'candidate', 'sector'], levels=[[],[],[]], codes=[[],[],[]])
        for ticid in self.sources.keys():
            cndts = self.data.loc[ticid].index
            sectors = self.data.loc[ticid].sectors.iloc[0]
            indx = indx.append(pd.MultiIndex.from_product([[ticid], cndts, sectors], names=['ticid','candidate','sector']))
        
        # Create empty dataframe with the multi-index constructed above
        new_df = pd.DataFrame(data=0, index=indx, columns=['cam', 'ccd', 'X_diff', 'X_err', 'Y_diff', 'Y_err', 'flag'])
        
        new_df['flag'] = ''
        
        if not rerun:
            if not preload:
                # Check if sector_data already exists and update the newly constructed dataframe with existing values
                if self.centroid is not None:
                    new_df = pd.concat([new_df, self.centroid])
                    new_df = new_df[~new_df.index.duplicated(keep='last')]   
            else:           
                # Update the dataframe with the preloaded values
                new_df = pd.concat([new_df, centroid_preloaded])
                new_df = new_df[~new_df.index.duplicated(keep='last')]
        
        self.centroid = new_df
        
        # Find the entries which are still zero
        to_fill = self.centroid.query('X_diff == 0 & X_err == 0 & Y_diff == 0 & Y_err == 0')
        num_targets = len(to_fill.index.unique('ticid'))
        
        if num_targets > 0:
            print(f'Running centroid data retrieval for {num_targets} targets')
            
            if self.multiprocessing > 1 and num_targets > 5:
                if num_targets < self.multiprocessing:
                    workers = num_targets
                else:
                    workers = self.multiprocessing
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    try:
                        if len(to_fill.index) < 20*workers:
                            index_split = np.array_split(to_fill.index, len(to_fill.index))
                        else:
                            index_split = np.array_split(to_fill.index, 20*workers)
                        futures = {ex.submit(self.multi_centroid, index_group): index_group for index_group in index_split}
                        
                        for future in as_completed(futures):
                            try:
                                filled, fails = future.result()
                                self.centroid = pd.concat([self.centroid, filled])
                                self.centroid = self.centroid[~self.centroid.index.duplicated(keep='last')]
                                if len(fails) > 0:
                                    for fail in fails:
                                        print(f'Exception "{fail[1]}" occur for: {fail[0]}')
                            except Exception as e:
                                group = futures[future]
                                print(f'Exception "{e}" occur for index group: {group}')
                    except KeyboardInterrupt:
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise ValueError('Keyboard interrupt')          
            else:
                filled, fails = self.multi_centroid(to_fill.index)
                self.centroid = pd.concat([self.centroid, filled])
                self.centroid = self.centroid[~self.centroid.index.duplicated(keep='last')]
                
                for fail in fails:
                    print(f'Exception {fail[1]} occur for index: {fail[0]}')
                                    
        if self.save_output:
            if save_overwrite:
                outfile = Path.cwd() / 'Output' / f'centroiddata_{save_overwrite}.csv'
            else:
                outfile = Path.cwd() / 'Output' / f'centroiddata_{self.save_suffix}.csv'
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.centroid.to_csv(outfile)
                
        self.centroid_data = True
                              
    def observed_centroid_offset(self, idx):
        targetid, cndt, sec = idx
        # Retrieve the sector data [per, t0, tdur, tdu23]
        event_data = self.sector_data.loc[(targetid, cndt, sec)]
        tc_per = event_data.new_per
        tc_t0 = event_data.sec_t0
        tc_tdur = event_data.sec_tdur
        tc_tdur23 = event_data.tdur23

        
        lcdir = self.data.loc[(targetid, cndt)].lcdir

        lc_file = list(lcdir.glob(f'*s{sec:04}*lc.fits'))[0]
            
        time, data_X, data_Y, cent_flag, cam, ccd = utils.load_spoc_centroid(lc_file,
                                                                                flatten=True, cut_outliers=5, trim=True,
                                                                                sectorstart=None, transitcut=False,
                                                                                tc_per=tc_per,
                                                                                tc_t0=tc_t0,
                                                                                tc_tdur=tc_tdur)
        
        if not cent_flag:
            X_diff, X_diff_err, Y_diff, Y_diff_err, cent_flag = utils.centroid_fitting(time, data_X, data_Y,
                                                                                        tc_per,
                                                                                        tc_t0,
                                                                                        tc_tdur, tc_tdur23, loss='huber')

            if  (np.isnan(X_diff) or np.isnan(Y_diff)) and not cent_flag:
                cent_flag = 'Nan from calculation'
        else:
            X_diff, X_diff_err, Y_diff, Y_diff_err = np.nan, np.nan, np.nan, np.nan
            
        return (targetid, cndt, sec, cam, ccd, X_diff, X_diff_err, Y_diff, Y_diff_err, cent_flag)
    
    def multi_centroid(self, indices):
        results = []
        fails = []
        for idx in indices:
            try:
                results.append(self.observed_centroid_offset(idx))
            except Exception as e:
                fails.append((idx, e))
                
        filled = pd.DataFrame([r for r in results], columns=['ticid', 'candidate', 'sector', 'cam', 'ccd', 'X_diff', 'X_err', 'Y_diff', 'Y_err', 'flag'])
        filled.set_index(['ticid', 'candidate', 'sector'], inplace=True)
        
        return filled, fails
        
        
    def estimate_flux_fractions(self, rerun=False):
        if not self.update_data:
            raise ValueError('Run update_transit_data first')
        
        for ticid in self.data.index.unique('ticid'):
            try:
                source_obj = self.sources[ticid]
            except KeyError:
                continue
            
            if source_obj.nearby_fractions is None or rerun:
                source_obj.nearby_fractions = pd.DataFrame(index=source_obj.nearby_data.index)
                
            sectors = self.data.loc[ticid].iloc[0].sectors

            # Check if sector fractions already present in the dataframe.
            # Difference will be only if the nearby_fractions dataframe was already existing in the source_obj data.
            # Allows to run the process only for new sectors
            already_run = np.array([int(x[1:]) for x in source_obj.nearby_fractions.columns])
            sectors = np.setdiff1d(np.array(sectors), already_run)                         
            for sec in sectors:
                file_loc = self.data.loc[ticid].iloc[0].lcdir
                filepath = list(file_loc.glob(f'*s{sec:04}*lc.fits'))[0]
                aperture_mask, centroid_mask, wcs, origin, cam, ccd, background_mask = utils.load_spoc_masks(filepath, background=True)
                source_obj.wcs[sec] = wcs
                source_obj.origin[sec] = origin
                source_obj.aperture_mask[sec] = aperture_mask
                source_obj.centroid_mask[sec] = centroid_mask
                source_obj.background_mask[sec] = background_mask
                source_obj.scc.loc[sec] = [cam, ccd]
                
                data = source_obj.nearby_data.copy()
                
                data['x'], data['y'] = wcs.all_world2pix(data.ra, data.dec, 0)
                
                # Test if target position from wcs is correctly in the aperture
                in_ap = utils.test_target_aperture(data.loc[ticid].x, data.loc[ticid].y, aperture_mask)
                
                if in_ap:
                    fractions, total_flux = utils.calc_flux_fractions(sec, cam, ccd, origin, data.x.values, data.y.values, data.Flux.values, aperture_mask)
                else:
                    fractions = np.zeros(len(data))
                    fractions[:] = np.nan
                    
                    total_flux = np.nan
                
                source_obj.totalflux[sec] = total_flux
                source_obj.totalmag_equivalent[sec] = 10 - 2.5*np.log10(total_flux/15000)
                source_obj.nearby_fractions[f'S{sec}'] = fractions
                
        # Save output to be reused
        if self.save_output:
            outfile = Path.cwd() / 'Output' / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
                
        self.flux_fractions = True

    def estimate_nearby_depths(self, rerun=False):
        if not self.flux_fractions:
            raise ValueError('Run estimate_flux_fractions first')
        
        for ticid in self.data.index.unique('ticid'):
            try:
                source_obj = self.sources[ticid]
            except KeyError:
                continue
            
            target_data = self.data.loc[ticid]
            target = self.sources[ticid]
            
            candidates = target_data.index.values
            indx = pd.MultiIndex.from_product([target.nearby_data.index.values, candidates], names=['ticid', 'candidate'])
            cols = [f'S{sec}' for sec in target_data.iloc[0].sectors]
            nearby_depths = pd.DataFrame(data=np.nan, index=indx, columns=cols)
            
            if not rerun:
                try:
                    nearby_depths = pd.concat([nearby_depths, target.nearby_depths.drop('Mean', axis=1)])
                    nearby_depths = nearby_depths[~nearby_depths.index.duplicated(keep='last')]
                except KeyError:
                    pass
            
            target.nearby_depths = nearby_depths

            # Fill in the depths per candidate
            for cndt in candidates:
                sub_depths = target.nearby_depths.query(f'candidate == {cndt}')
                null_sectors = sub_depths.loc[:, sub_depths.isnull().all()].columns
    
                for sec in null_sectors:
                    sec_num = int(sec[1:])

                    cndt_sec_depth = self.sector_data.loc[(ticid, cndt, sec_num), 'sec_depth']
                    if cndt_sec_depth < 50:
                        cndt_sec_depth = 0

                    f_target = target.nearby_fractions.loc[ticid, sec]
                    f_nearby = target.nearby_fractions[sec].values
                    depths = utils.nearby_depth(cndt_sec_depth, f_target, f_nearby)

                    target.nearby_depths.loc[target.nearby_depths.index.get_level_values('candidate') == cndt, sec] = depths
            
            target.nearby_depths['Mean'] = target.nearby_depths.replace(0, np.nan).mean(axis=1)
            
        self.estimate_depths = True
              
    def cut_faintstars(self, max_eclipsedepth=1.0, max_transitdepth=0.02, centroid_thresh=1e-4, rerun=False):
        """
        Take result of wide search for nearby targets and remove those that are too faint/distant/excluded by centroid
        
        max transit depth and eclipse depth should be set based on simulation input to scenarios
        
        CENTROID BITS TO BE REINCLUDED ONCE CENTROID DATA IN PLACE
        
        centroid thresh is a 2d gaussian mahalanobis distance equivalent. 0.003 corresponds to 99.7% of the distribution being 
        closer than the sample, the 3 sigma equivalent. 3e-7 is the 5 sigma equivalent, chosen because our errors might be
        underestimated
        """

        if not self.centroid_data:
            raise ValueError('Run generate_centroiddata first!')
        
        if not self.flux_fractions:
            raise ValueError('Run nearby_flux_fractions first!')
         
        for targetid in self.data.index.unique('ticid'):
            target_data = self.data.loc[targetid]
            try:
                source_obj = self.sources[targetid]
            except KeyError:
                continue
            
            sectors = target_data.iloc[0].sectors
            
            if not rerun:
                sectors_out = np.setdiff1d(sectors, source_obj.cent_out.index)
            else:
                sectors_out = sectors
            
            for sec in sectors_out:
                
                cam, ccd = source_obj.scc.loc[sec]
                origin = source_obj.origin[sec]
                wcs = source_obj.wcs[sec]
                centroid_mask = source_obj.centroid_mask[sec]
                
                data = source_obj.nearby_data.copy()
                data['x'], data['y'] = wcs.all_world2pix(data.ra, data.dec, 0)
                
                in_ap = utils.test_target_aperture(data.iloc[0].x, data.iloc[0].y, centroid_mask)
                
                if in_ap:
                    fractions = utils.prf_fractions(sec, cam, ccd, origin, data.x.values, data.y.values, centroid_mask)
                    source_obj.cent_fractions[sec] = fractions
                    cent_x, cent_y = utils.centroid_shift(centroid_mask, data.Flux.values, fractions)
                else:
                    cent_x, cent_y = np.nan, np.nan
                
                source_obj.cent_out.loc[sec] = cent_x, cent_y

            cent_in = pd.DataFrame(data=0, 
                                   index=pd.MultiIndex.from_product([source_obj.nearby_data.index, target_data.index, sectors], names=['ticid', 'candidate', 'sector']), 
                                   columns=['X', 'Y', 'X+', 'Y+', 'X-', 'Y-'])
            if not rerun:
                cent_in = pd.concat([cent_in, source_obj.cent_in])
                cent_in = cent_in[~cent_in.index.duplicated(keep='last')]
                
            source_obj.cent_in = cent_in
            
            # model_centroid = pd.DataFrame(data=0,index=cent_in.index, columns=['X_diff', 'X_err', 'Y_diff', 'Y_err'])
            # model_centroid.update(source_obj.model_centroid)
            # source_obj.model_centroid = model_centroid
                                                
            to_fill = source_obj.cent_in.loc[(cent_in == 0).all(axis=1)]
            if len(to_fill) > 0:
                for sec in to_fill.index.unique('sector'):
                    
                    cam, ccd = source_obj.scc.loc[sec]
                    origin = source_obj.origin[sec]
                    wcs = source_obj.wcs[sec]
                    centroid_mask = source_obj.centroid_mask[sec]
                    
                    data = source_obj.nearby_data.copy()
                    data['x'], data['y'] = wcs.all_world2pix(data.ra, data.dec, 0)
                    
                    in_ap = utils.test_target_aperture(data.iloc[0].x, data.iloc[0].y, centroid_mask)
                    
                    for cndt in to_fill.query(f'sector == {sec}').index.unique('candidate'):
                        if self.centroid.loc[targetid, cndt, sec].isna().any():
                            source_obj.cent_in.loc[cent_in.query(f'candidate == {cndt} & sector == {sec}').index] = np.nan
                            continue
                        for ticid in source_obj.nearby_data.index.unique('ticid'):
                            depth = source_obj.nearby_depths.loc[(ticid, cndt), f'S{sec}']*1e-6
                            mean_depth = source_obj.nearby_depths.loc[(ticid, cndt), f'Mean']*1e-6
                            depth = mean_depth
                            if not in_ap:
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            elif depth == 0.0 or np.isnan(depth):
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            elif depth*0.9 > max_eclipsedepth:
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            else:
                                flux_fractions = source_obj.cent_fractions[sec]
                                fluxes = data['Flux'].copy()
                                
                                fluxes.loc[ticid] *= (1-depth) 
                                
                                X, Y = utils.centroid_shift(centroid_mask, fluxes.values, flux_fractions)
                                
                                fluxes = data['Flux'].copy()
                                fluxes.loc[ticid] *= (1-depth*1.1) 
                                
                                X_plus, Y_plus = utils.centroid_shift(centroid_mask, fluxes.values, flux_fractions)
                                
                                fluxes = data['Flux'].copy()
                                fluxes.loc[ticid] *= (1-depth*0.9) 
                                
                                X_minus, Y_minus = utils.centroid_shift(centroid_mask, fluxes.values, flux_fractions)

                                source_obj.cent_in.loc[ticid, cndt, sec] = [X, Y, X_plus, Y_plus, X_minus, Y_minus]

                # implicit assumption that unseen stars are too faint to affect the calculation
                model_diff = pd.DataFrame()
                model_diff['X_diff'] = source_obj.cent_in['X'] - source_obj.cent_out['X']
                model_diff['Y_diff'] = source_obj.cent_in['Y'] - source_obj.cent_out['Y']
                model_diff['X_err1'] = source_obj.cent_in['X'] - source_obj.cent_in['X-']
                model_diff['X_err2'] = source_obj.cent_in['X+'] - source_obj.cent_in['X']
                model_diff['Y_err1'] = source_obj.cent_in['Y'] - source_obj.cent_in['Y-']
                model_diff['Y_err2'] = source_obj.cent_in['Y+'] - source_obj.cent_in['Y']
                model_diff['X_err'] = model_diff[['X_err1', 'X_err2']].max(axis=1)
                model_diff['Y_err'] = model_diff[['Y_err1', 'Y_err2']].max(axis=1)
                                
                source_obj.model_centroid = model_diff[['X_diff', 'X_err', 'Y_diff', 'Y_err']]
                
                # Add the observed centroid shift, per sector, to ease subsequent calculations
                model_diff = model_diff.join(self.centroid.loc[targetid], on=['candidate', 'sector'], rsuffix='_obs')
                
                model_diff['Probability'] = model_diff.apply(lambda x: utils.calc_centroid_probability_all(x['X_diff_obs'], x['Y_diff_obs'], 
                                                                                                            x['X_err_obs'], x['Y_err_obs'], 
                                                                                                            x['X_diff'], x['Y_diff'], 
                                                                                                            x['X_err'], x['Y_err']), axis=1)
                
                # Compute the sum of the probabilities for all candidates per sector
                model_diff = model_diff.join(model_diff.groupby(['candidate', 'sector']).agg(Prob_Sum=('Probability', sum)), on=['candidate', 'sector'])
                
                # model_diff = model_diff.join(model_diff.groupby(['candidate', 'sector']).agg(MaxProb=('Probability', max)), on=['candidate', 'sector'])
                # model_diff.loc[model_diff['MaxProb'] < 0.01, 'Probability'] = np.nan
                
                # Compute the normalised probability for each candidate per sector
                model_diff['Norm_Probability'] = model_diff['Probability'] / model_diff['Prob_Sum']
                
                source_obj.model_prob = model_diff[['Probability', 'Prob_Sum', 'Norm_Probability']]
                
                # model_diff.loc[model_diff['Norm_Probability'].isna(), 'Norm_Probability'] = 0
                
                prob_centroid = pd.DataFrame(index=pd.MultiIndex.from_product([source_obj.nearby_data.index, target_data.index], names=['ticid', 'candidate']))
                prob_centroid = prob_centroid.join(model_diff.groupby(['ticid', 'candidate']).agg(NormProbMean=('Norm_Probability','mean')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_diff.groupby(['ticid', 'candidate']).agg(NormProbMedian=('Norm_Probability','median')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(prob_centroid.groupby(['candidate']).agg(NormMeanSum=('NormProbMean','sum')), on=['candidate'])
                prob_centroid = prob_centroid.join(prob_centroid.groupby(['candidate']).agg(NormMedianSum=('NormProbMedian','sum')), on=['candidate'])
                prob_centroid = prob_centroid.join(model_diff.groupby(['ticid', 'candidate']).agg(MaxProbability=('Probability','max')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_diff.groupby(['ticid', 'candidate']).agg(MeanProbability=('Probability','mean')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_diff.groupby(['ticid', 'candidate']).agg(MedianProbability=('Probability','median')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(prob_centroid.groupby(['candidate']).agg(MeanSum=('MeanProbability','sum')), on=['candidate'])
                prob_centroid = prob_centroid.join(prob_centroid.groupby(['candidate']).agg(MedianSum=('MedianProbability','sum')), on=['candidate'])
                prob_centroid['Norm_Mean'] = prob_centroid['MeanProbability'] / prob_centroid['MeanSum']
                prob_centroid['Norm_Median'] = prob_centroid['MedianProbability'] / prob_centroid['MedianSum']
                prob_centroid['Probability'] = prob_centroid['NormProbMean'] / prob_centroid['NormMeanSum']
                prob_centroid['ProbabilityAlt'] = prob_centroid['NormProbMedian'] / prob_centroid['NormMedianSum']
                prob_centroid.drop(['MeanSum','MedianSum','NormMeanSum', 'NormMedianSum'], axis=1, inplace=True)
                            
                # Change nan probabilities to 0
                # prob_centroid[prob_centroid.isnull().any(axis=1)] = 0
                
                # Store probability centroid to the source
                source_obj.prob_centroid = prob_centroid
                            
                # Assess the suitability of each nearby as the host of the event
                nearby_assessment = pd.DataFrame(index=pd.MultiIndex.from_product([source_obj.nearby_data.index, target_data.index], names=['ticid', 'candidate']))
                nearby_assessment['Possible'] = True
                nearby_assessment['Rejection Reason'] = '' 
            
                # Check for zero depth and hence zero flux in aperture
                nearby_assessment.loc[source_obj.nearby_depths['Mean'] == 0] = False, 'Zero flux in aperture'
                # Check for nan Mean depth
                nearby_assessment.loc[source_obj.nearby_depths['Mean'].isna()] = False, 'Nan depth'
                # Check for depth above max eclipse depth, with some room for error
                nearby_assessment[source_obj.nearby_depths['Mean']*1e-6*0.9 > max_eclipsedepth] = False, 'Eclipse depth above max'
                            
                # For the rest check the centroid probability
                mask = nearby_assessment['Possible'] == True
                # Below threshold
                nearby_assessment.loc[source_obj.prob_centroid[mask].query(f'Probability < {centroid_thresh}').index] = False, 'Centroid probability below threshold'
                # Nan
                nearby_assessment.loc[source_obj.prob_centroid[mask].query(f'Probability != Probability').index] = False, 'Nan centroid probability'
                
                # Finally, check if the nearby is a potential planetary scenario
                nearby_assessment['Planet'] = True
                # Check for depth above max planet transit depth, with some room for error
                nearby_assessment.loc[source_obj.nearby_depths['Mean']*1e-6*0.9 > max_transitdepth, 'Planet'] = False
                # Or for nan mean depth
                nearby_assessment.loc[source_obj.nearby_depths['Mean'].isna(), 'Planet'] = False
                
                # Store assessment in source_obj
                source_obj.nearby_assessment = nearby_assessment
                
            cent_prob = source_obj.prob_centroid.loc[source_obj.nearby_assessment.query(f'Possible == True or ticid == {targetid}').index].copy()
            try:
                disp = target_data['tfop_disp']
                if pd.isna(disp).any():
                    disp = target_data['tess_disp']
                disp.name = 'disp'
            except KeyError:
                disp = pd.DataFrame(data= ['TCE'] * len(target_data.index), index=target_data.index, columns=['disp'])
                
            cent_prob = cent_prob.join(disp)
            cent_prob.loc[cent_prob.query(f'ticid != {targetid}').index, 'disp'] = 'NFP'
            cent_prob['target'] = targetid
            cent_prob.reset_index(inplace=True)
            cent_prob.rename(columns={'ticid':'source'}, inplace=True)
            cent_prob.set_index(['target', 'candidate', 'source'], inplace=True)
            
            self.probabilities = pd.concat([self.probabilities, cent_prob])
        self.probabilities = self.probabilities[~self.probabilities.index.duplicated(keep='last')]    
        # Save output to be reused
        if self.save_output:
            outfile = Path.cwd() / 'Output' / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
            outfile = Path.cwd() / 'Output' / f'Probabilities_{self.save_suffix}.csv'
            self.probabilities.to_csv(outfile)


class Source(object):

    def __init__(self, tic, coords=(), mags={}, lclass=None, star_rad=None, star_mass=None, plx=None, teff=None):
        """
        Datastore for a flux source
        """
        self.TIC = tic # Target TIC ID
        self.coords = coords 
        self.mags = mags
        self.scc = pd.DataFrame(columns=['sector', 'cam', 'ccd']).set_index('sector')
        self.wcs = {}
        self.origin = {}
        self.aperture_mask = {}
        self.centroid_mask = {}
        self.background_mask = {}
        self.nearby_data = None
        self.nearby_fractions = None
        self.nearby_depths = pd.DataFrame()
        self.totalflux = {}
        self.totalmag_equivalent = {}
        self.cent_fractions = {}
        self.cent_out = pd.DataFrame(columns=['sector', 'X', 'Y']).set_index('sector') # Model centroid out-of-transit
        self.cent_in = pd.DataFrame() # Model centroid in-transit 
        self.model_centroid = pd.DataFrame()
        self.prob_centroid = pd.DataFrame()
        self.nearby_assessment = None
