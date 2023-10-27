import numpy as np
import pandas as pd
from pathlib import Path
import os

import astropy.units as u

import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from CandidateSet import utils


class CandidateSet(object):

    def __init__(self, infile, infile_type='default', lc_dir='default', per_lim=None, depth_lim=None, multiprocessing=0, save_output=False, save_suffix=None, load_suffix=None, plot_centroid=False):
        """
        Load in a set of candidates with their transit parameters [period, epoch, depth]. Sets up the environment for running the positional probabilitiy generation.         
        
        Parameters
        infile - path to input file 
        infile_type - options: default/archive/exofop, allows for loading in data from specific databases or the default loading format.
        lc_dir - either path to lighcurve directory or set as default.
        per_lim - set maximum period limit for candidates to be processed. Candidates with period longer than maximum will be skipped.
        depth_lim - set minimum depth limit for candidates. Candidates with depth less than the minimum will be skipped.
        multiprocessing - set maximum number of workers (set 0 for no multiprocessing and 1 to use all available processors)
        save_output - True/False or 'full'. True outputs only a csv file with the Positional Probabilities. 'full' enables output for all data generation functions and extends the probability output.
        save_suffix - Suffix for the filenames of all saved data.
        load_suffix - Suffix for loading previously saved data.
        plot_centroid - True/False. Create plots when fitting the trapezium transit model to the centroid data.
        """

        if infile_type == 'exofop':
            self.data = utils.load_exofop_toi(infile, lc_dir, per_lim=per_lim, depth_lim=depth_lim)
        elif infile_type == 'archive':
            self.data = utils.load_archive_toi(infile, lc_dir, per_lim=per_lim, depth_lim=depth_lim)
        elif infile_type == 'default':
            self.data = utils.load_default(infile, lc_dir, per_lim=per_lim, depth_lim=depth_lim)
        else:
            raise ValueError('Infile type must be set as one of: default/archive/exofop')
        
        # Data containers    
        self.sources = {} # Source objects
        self.sector_data = None # Per sector transit data
        self.centroid = None # Observed centroid offsets
        self.probabilities = pd.DataFrame() # Positional Probabilities
        self.assessment = pd.DataFrame() # Assessment for all nearby sources

        self.find_stars = False
        self.sectordata = False
        self.centroiddata = False 
        self.flux_fractions = False
        self.estimate_depths = False
        self.possible_sources = False
        
        if multiprocessing == 1:
            multiprocessing = os.cpu_count()
            
        self.multiprocessing = multiprocessing
        
        if isinstance(save_output, str):
            save_output = save_output.lower()
            
        self.save_output = save_output
        
        if save_output is not False:   
            if save_suffix:
                self.save_suffix = save_suffix
            else:
                self.save_suffix = datetime.today().strftime('%d%m%Y_%H%M%S')
        
            # Define and create output directory if it does not exist       
            self.output = Path(__file__).resolve().parents[1] / 'Output'
            self.output.mkdir(exist_ok=True)
        
        self.load_suffix = load_suffix
        
        self.plot_centroid = plot_centroid
        # Define and create the directory to save the centroid plots. 
        # Individual folders for each target will be created later within this directory
        if self.plot_centroid:
            outfile = Path(__file__).resolve().parents[1] / 'Output' / 'Plots'
            outfile.mkdir(exist_ok=True, parents=True)
        
    
    def generate_sources(self, infile=None, rerun=False):
        """
        Retrieve stellar characteristics from the TIC for each target star 
        and generate the source object data-containeres that will be used to run the pipeline. 
        Identify nearby TIC stars for each candidate, up to 8.5 ΔTmag.
        
        Parameters:
        infile - Overwrite the load suffix specified in class creation to load a user specified file.
        rerun - force the nearby star indentification process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if sources already exist
        """
        # Get the unique ticids. Multiple candidates on one source will result to only one source created.
        targets = self.data.index.unique('ticid')

        # Load in data from file, if provided at class initialisation
        preload = False
        if (self.load_suffix or infile) and not rerun and not self.find_stars:
            if infile:
                infile = Path(infile)
            else:
                infile = self.output / f'sources_{self.load_suffix}.pkl'
            print('Loading from ' + str(infile))
            try:
                with open(infile, 'rb') as f:
                    self.sources = pickle.load(f)

                # Compare loaded sources ids with the ids of the targets in the data to find any that might be missing
                targets = np.setdiff1d(targets, list(self.sources.keys()), assume_unique=True)
                preload = True
            except Exception as e:
                # Handle failure to load data
                print(e)
                print('Error loading saved sources, recreating...')
                pass
        
        # If the function is called multiple times during execution, this allows for reusing the existing sources.
        # Usefull in case of disconnections when retrieving data from MAST
        if not preload and not rerun and len(self.sources.keys()) > 0:
            targets = np.setdiff1d(targets, list(self.sources.keys()), assume_unique=True)
            print('Existing sources found and are being reused')
                
        if len(targets) > 0:
            print(f'Retrieving data from MAST for {len(targets)} targets')
            source_fail = []
            duplicate = []
            artifact = []
            non_gaia = []
            
            # Run the identification process
            if self.multiprocessing > 1 and len(targets) > 10:
                # Multithread if requested and there are enough targets to be worth it
                # Set the number of mutlithreading workers. Maximum of 20 to not overload MAST with requests
                workers = np.min((self.multiprocessing*4, 20)) 
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    try:
                        futures = {ex.submit(self.find_TIC_stars, targetid): targetid for targetid in targets}
                        
                        # Retrieve results as they are returned
                        for future in as_completed(futures):
                            targetid = futures[future] 
                            try:
                                source_obj, fail = future.result()
                                if source_obj and fail is None:
                                    # Store the source object in the sources class dictionary if it was successfully created 
                                    self.sources[targetid] = source_obj
                                else:
                                    # Handle the failure cases 
                                    if fail == 'DUPLICATE':
                                        print(f'{targetid} skipped due to being flagged as a duplicate')
                                        duplicate.append(targetid)
                                    elif fail == 'ARTIFACT':
                                        print(f'{targetid} skipped due to being flagged as an artifact')
                                        artifact.append(targetid)
                                    elif fail == 'NonGaia':
                                        self.sources[targetid] = source_obj
                                        print(f'{targetid} flagged as it has no Gaia ID')
                                        non_gaia.append(targetid)
                                    else:
                                        print(f'{targetid} source failed to be created.')
                                        source_fail.append(targetid)
                                    
                            except Exception as e:
                                # Handle uncaught exceptions
                                print(f'Source not created for {targetid} due to exception: {e}')
                                source_fail.append(targetid)
           
                    except KeyboardInterrupt:
                        # Attempt to shutdown the multithreaded workload if interrupted
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise ValueError('Keyboard interrupt')
            else:
                # Single threaded workload
                for targetid in targets:
                    source_obj, fail = self.find_TIC_stars(targetid)
                    if source_obj and fail is None:
                        # Store the successfully created sources
                        self.sources[targetid] = source_obj
                    else:
                        # Handle the failure cases 
                        if fail == 'DUPLICATE':
                            print(f'{targetid} skipped due to being flagged as a duplicate')
                            duplicate.append(targetid)
                        elif fail == 'ARTIFACT':
                            print(f'{targetid} skipped due to being flagged as an artifact')
                            artifact.append(targetid)
                        elif fail == 'NonGaia':
                            self.sources[targetid] = source_obj
                            print(f'{targetid} flagged as it has no Gaia ID')
                            non_gaia.append(targetid)
                        else:
                            print(f'{targetid} source failed to be created.')
                            source_fail.append(targetid)
            
            # Handle duplicate target sources
            
            self.data.drop(duplicate, axis=0, level=0, inplace=True)
            # Remove artifact sources
            self.data.drop(artifact, axis=0, level=0, inplace=True)
            # Output completion log
            print('Source identification and object creation completed.')
            if len(duplicate) > 0:
                print(f'{len(duplicate)} duplicate target source(s) removed:', duplicate)
            if len(artifact) > 0:
                print(f'{len(artifact)} artifact target source(s) removed:', artifact)
            if len(non_gaia) > 0:
                print(f'{len(non_gaia)} non Gaia target source(s) flagged:', non_gaia)  
            if len(source_fail) > 0:   
                print(f'{len(source_fail)} source(s) failed to be created:', source_fail)
            else:
                # Mark the process as completed
                self.find_stars = True

        # Save output to be reused if specified when class was initialised
        if self.save_output == 'full':
            outfile = self.output / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
                
    @staticmethod            
    def find_TIC_stars(targetid):
        '''
        Function for retrieving the stellar characteristics of target stars and their nearby identified sources.
        Separated to aid mutlti-threaded runs.
        
        Parameters:
        targetid - the TIC id of the target star
        '''
        source_obj = None 
        fail = None # Handle for failure cases
        s_rad = 168 * u.arcsec  # 8 TESS pixels
        
        # Retrieve the data for the target and its nearby sources
        sources_data = utils.query_tic_by_id(targetid, s_rad)

        # Handle failure to find target
        if sources_data is None:
            return source_obj, fail
        
        # Select the target data
        target_data = sources_data.query(f'ticid == {targetid}').iloc[0]
        # Check if target is flagged as duplicate or artifact
        if target_data['disposition'] == 'ARTIFACT' or target_data['disposition'] == 'DUPLICATE':
            fail = target_data['disposition']
            return source_obj, fail
        if pd.isna(target_data['Gaia_id']):
            fail = 'NonGaia'
        
        # Remove artifact or duplicate nearby sources
        sources_data.drop(sources_data.query('disposition == "DUPLICATE" or disposition == "ARTIFACT"').index, inplace=True)
        
        # Identify sources with no Gaia entry
        idx_non_gaia = pd.isna(sources_data['Gaia_id'])
        
        # Set the proper motion of non Gaia sources to 0
        sources_data.loc[idx_non_gaia, 'pmra'] = 0
        sources_data.loc[idx_non_gaia, 'pmdec'] = 0
        
        # Set nan proper motion to 0, to ease subsequent calculations
        idx_na = pd.isna(sources_data[['pmra', 'pmdec']]).any(axis=1)
        sources_data.loc[idx_na, 'pmra'] = 0
        sources_data.loc[idx_na, 'pmdec'] = 0

        # Set up a source object for the target   
        # Retrieve the TESS, GAIA and V magnitudes
        mags = {'TESS': (target_data['Tmag'], target_data['e_Tmag']),
                'G': (target_data['Gmag'], target_data['e_Gmag']),
                'V': (target_data['Vmag'], target_data['e_Vmag'])}
        
        # Create the source object
        source_obj = Source(tic = targetid,
                            coords = (target_data['ra'], target_data['dec']),
                            pm = (float(target_data['pmra']), float(target_data['pmdec'])), 
                            mags = mags,
                            star_rad = target_data['rad'],
                            star_mass = target_data['mass'],
                            teff = target_data['Teff'])
        

        # Remove faint sources with dTmag greater than 10
        tmag_limit = source_obj.mags['TESS'][0] + 10
        sources_data.query(f'Tmag <= {tmag_limit}', inplace=True)
        sources_data.set_index('ticid', inplace=True)
            
        # Use the Tmag to calculate the expected flux counts observed by each source
        sources_data['Flux'] = 15000 * 10 ** (-0.4 * (sources_data['Tmag'] - 10))
            
        # Ensure that the target is always first in the dataframe
        sources_data['order'] = np.arange(len(sources_data))  + 1
        sources_data.loc[targetid, 'order'] = 0
        sources_data.sort_values('order', inplace=True)
        sources_data.drop('order', axis=1, inplace=True)
            
        # Store the nearby data into the source object
        source_obj.nearby_data = sources_data
        
        return source_obj, fail


    def generate_per_sector_data(self, infile=None, rerun=False):
        """
        Determine the per sector depth, transit duration and the duration without ingress/egrees.
        
        Parameters:
        infile - Overwrite the load suffix specified in class creation to load a user specified file.
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if sector_data already exist
        """
        
        # Check if the stars and the nearbies have been identified and the sources dictionary has been populated. 
        # Raise an error if not.
        if not self.find_stars:
            raise ValueError('Run find_tic_stars first')
        
        # Load in data from file, if provided at class initialisation or when the function was called.
        preload = False
        if (self.load_suffix or infile) and not rerun and not self.sectordata:
            if infile:
                infile = Path(infile)
            else:
                infile = self.output / f'sectordata_{self.load_suffix}.csv'
            print('Loading from ' + str(infile))
            try:
                data_preloaded = pd.read_csv(infile).set_index(['ticid', 'candidate', 'sector'])
                preload = True
            except Exception as e:
                # Handle failure to load existing data
                print(e)
                print('Error loading existing sector data, recreating...')
        
        # Create empty multi-index for the ticid, candidate and sector combinations. 
        indx = pd.MultiIndex(names=['ticid', 'candidate', 'sector'], levels=[[],[],[]], codes=[[],[],[]])
        for ticid in self.sources.keys():
            cndts = self.data.loc[ticid].index
            sectors = self.data.loc[ticid].sectors.iloc[0]
            indx = indx.append(pd.MultiIndex.from_product([[ticid], cndts, sectors], names=['ticid','candidate','sector']))
        
        # Create new dataframe with the multi-index constructed above, data initialised with 0
        new_df = pd.DataFrame(data=0, index=indx, columns=['t0', 'per', 'sec_tdur', 'sec_tdur23', 'sec_depth'])
        
        if not rerun:
            if not preload:
                # Check if sector_data already exists and update the newly constructed dataframe with existing values
                if self.sector_data is not None:
                    new_df = pd.concat([new_df, self.sector_data])
                    new_df = new_df[~new_df.index.duplicated(keep='last')]   
            else:           
                # Update the dataframe with the preloaded values
                new_df = pd.concat([new_df, data_preloaded])
                new_df = new_df[~new_df.index.duplicated(keep='last')]
    
        self.sector_data = new_df
        
        # Find the entries which are still zero. Those will need to be filled.
        to_fill = self.sector_data.query('t0 == 0')
        ticids = to_fill.index.unique('ticid')
        num_targets = len(ticids)
        
        if num_targets > 0:
            print(f'Running sector_data update for {num_targets} targets')
            if self.multiprocessing > 1 and num_targets > 1:
                # Run multi-processed if requested at class initialisation and there are enough targets to be worth it
                if num_targets < self.multiprocessing:
                    # If there are less targets than the specified number of cores, set the number of workers accordingly
                    workers = num_targets
                else:
                    workers = self.multiprocessing
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    try:
                        # Split the data into chunks based on the number of workers, to aid multiprocessing performance
                        factor = 20
                        while num_targets < 5*factor*workers:
                            factor -= 1
                            if factor == 1:
                                break
                        ticid_split = np.array_split(ticids, factor*workers)
                        #print(len(ticid_split))  
                        
                        # Run the multiprocessed job, in a chunk based approach
                        futures = {ex.submit(self.multi_target_data, ticid_group): ticid_group for ticid_group in ticid_split}
                        
                        for future in as_completed(futures):
                            # Handle the results as they are completed. 
                            try:
                                filled, fails = future.result()
                                self.sector_data = pd.concat([self.sector_data, filled])
                                self.sector_data = self.sector_data[~self.sector_data.index.duplicated(keep='last')]
                                if len(fails) > 0:
                                    # For individual exceptions on a target, explicitly caught and handled in the code
                                    for fail in fails:
                                        print(f'Exception {fail[1]} occur with ticid: {fail[0]}')
                            except Exception as e:
                                # For exceptions that were not caught and handled in the code, which lead to failure for the whole group of ticids
                                group = futures[future]
                                print(f'Exception {e} occur with ticid group: {group}')
                    except KeyboardInterrupt:
                        # Attempt to shutdown multi-processed work, if interrupted
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise ValueError('Keyboard interrupt')            
            else:
                # Run on a single core
                ticid_split = np.array_split(ticids, int(np.ceil(len(ticids)/10)))
                for ticid_group in ticid_split:
                    filled, fail = self.multi_target_data(ticid_group)
                    
                    self.sector_data = pd.concat([self.sector_data, filled])
                    self.sector_data = self.sector_data[~self.sector_data.index.duplicated(keep='last')]
                
                    for ticid, e in fail:
                        print(f'Exception {e} occur with ticid: {ticid}')
            
        # Save data if specified when class was initialised                                      
        if self.save_output == 'full':
            outfile = self.output / f'sectordata_{self.save_suffix}.csv'
            self.sector_data.to_csv(outfile)
        
        # Mark the process as completed      
        self.sectordata = True
             
                
    def multi_target_data(self, ticids):
        """
        Runs the sector_data generation for a group of targets. 
        
        Parameters:
        ticids - An array of target TIC IDs
        """
        results = []
        fails = []
        # Loop through the ticids to produce dataframes with ticid/canidate/sector entries
        for ticid in ticids:
            try:
                # Store the dataframes in a list
                results.append(self.per_sector_data(ticid, self.data.loc[ticid]))
            except Exception as e:
                # Store the failure cases and their associated exception
                fails.append((ticid, e))
                
        # Concatenate the group dataframes into one
        try:
            filled = pd.concat(results, ignore_index=True)
            filled.set_index(['ticid', 'candidate', 'sector'], inplace=True)
        except ValueError:
            # Incase that results is empty due to all targets failing
            filled = None
        
        return filled, fails
    
    
    def per_sector_data(self, ticid, initial_data):
        """
        Determine per sector transit duration, non-ingress/egress duration and depth for a single candidate by fitting a trapezium model to the TESS lightcurve 
        
        Parameters:
        ticid - target TIC ID
        initial_data - dataframe entry with the target's initial data
        """

        # Retrieve the lc location, sectors and candidates from the provided data
        lcdir = initial_data.iloc[0].lcdir
        sectors = sorted(initial_data.iloc[0].sectors)
        candidates = initial_data.index.to_numpy()
        
        # Empty arrays to stitch the sector lcs into one
        fluxes = np.array([])
        times = np.array([])
        errors = np.array([])
        
        # Load and store the detrended and normalised lightcurves per sector of observation
        lcs = {}
        hdus = {}
        sec_error = []
        for sec in sectors:
            lcfile = list(lcdir.glob(f'*{sec:04}*lc.fits'))[0]
            lc = utils.load_spoc_lc(lcfile, flatten=True, transitcut=True,
                                    tc_per=initial_data.per.values, 
                                    tc_t0=initial_data.t0.values,
                                    tc_tdur=initial_data.tdur.values)
            
            # Set the flux MAD as the error
            lc['error'] = np.full_like(lc['flux'], utils.MAD(lc['flux']))
            sec_error.append(lc['error'][0])
            lcs[sec] = lc
            
            fluxes = np.concatenate([fluxes, lc['flux']])
            times = np.concatenate([times, lc['time']])
            errors = np.concatenate([errors, lc['error']])

        lc = {'flux':fluxes,
              'time':times,
              'error':errors}
        sector_data = [] # List to store the results produced per candidate 
        
        # Process each candidate separately
        for cndt in candidates:
            # Create empty dataframe to store the per sector data
            cols = ['ticid', 'candidate', 'sector', 't0', 'per', 'sec_tdur', 'sec_tdur23', 'sec_depth', 'sec_flux', 'sec_time']
            df = pd.DataFrame(index=range(len(sectors)), columns=cols)
            # Retrieve provided t0, per, tdur, depth per candidate
            cndt_t0 = initial_data.loc[cndt, 't0']
            cndt_per = initial_data.loc[cndt, 'per']
            cndt_tdur = initial_data.loc[cndt, 'tdur']
            cndt_depth = initial_data.loc[cndt, 'depth']
            
            # Fill in the dataframe with the existing values
            df['ticid'] = ticid
            df['candidate'] = cndt
            df['sector'] = sectors
            df['t0'] = cndt_t0
            df['per'] = cndt_per
            
            # Calculate transit parameters based on the given period
            sec_tdur = []
            sec_tdur23 = []
            sec_depth = []
            
            for sec in sectors:
                fit_tdur, fit_tdur23, fit_depth = utils.transit_params(lcs[sec], cndt_t0, cndt_per, cndt_tdur, cndt_depth)
                
                sec_tdur.append(fit_tdur)
                sec_tdur23.append(fit_tdur23)
                sec_depth.append(fit_depth)
                                           

            # Store the sector parameters in the dataframe
            df['sec_tdur'] = sec_tdur
            df['sec_tdur23'] = sec_tdur23
            df['sec_depth'] = sec_depth
            
            # Append the dataframe to the list
            sector_data.append(df)
        
        # Concatenate the candidate datraframes in one
        sector_data = pd.concat(sector_data, ignore_index=True)
        
        # Add the median sector flux and the observation start time to the dataframe
        for sec in sectors:
            idx = sector_data.query(f'sector == {sec}').index
            sector_data.loc[idx, 'sec_flux'] = lcs[sec]['median']
            sector_data.loc[idx, 'sec_time'] = lcs[sec]['time'][0]
                                         
        return sector_data
    
        
    def generate_centroiddata(self, infile=None, rerun=False):
        """
        Calculates the centroid offset in tranist for the dataset
        
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if centroid_data already exist
        """
        # Check if the per sector data generation process has been completed before running this process, as it makes use of the sector data. 
        # Raise an error if not.
        if not self.sectordata:
            raise ValueError('Run find_tic_stars first')
        
        preload = False
        if (self.load_suffix or infile) and not rerun and not self.centroiddata:
            if infile:
                infile = Path(infile)
            else:
                infile = self.output / f'centroiddata_{self.load_suffix}.csv'
                
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
        
        # Create empty dataframe with the multi-index constructed above. Initialise data entries with 0
        new_df = pd.DataFrame(data=0, index=indx, columns=['cam', 'ccd', 'X_diff', 'X_err', 'Y_diff', 'Y_err', 'flag'])
        
        # Set the flag entries to empty string
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
            
            if self.multiprocessing > 1 and num_targets > 1:
                # Run multi-processed if requested at class initialisation and there are enough targets to be worth it
                if num_targets < self.multiprocessing:
                    # If there are less targets than the specified number of cores, set the number of workers accordingly
                    workers = num_targets
                else:
                    workers = self.multiprocessing
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    try:
                        # Split the data into chunks based on the number of workers, to aid multiprocessing performance
                        factor = 20
                        while len(to_fill.index) < 5*factor*workers:
                            factor -= 1
                            if factor == 1:
                                break
                        index_split = np.array_split(to_fill.index, factor*workers)
                        
                        # Run the mutliprocessed job, in a chunk based approach
                        # Compared to the sector_data process, the chunks include target, candidate, sector entries, not just ids
                        futures = {ex.submit(self.multi_centroid, index_group): index_group for index_group in index_split}
                        
                        for future in as_completed(futures):
                            # Handle the results as they are completed. 
                            try:
                                filled, fails = future.result()
                                self.centroid = pd.concat([self.centroid, filled])
                                self.centroid = self.centroid[~self.centroid.index.duplicated(keep='last')]
                                if len(fails) > 0:
                                    # Individual exceptions, explicitly caught, handled and reported in the code
                                    for fail in fails:
                                        print(f'Exception "{fail[1]}" occur for: {fail[0]}')
                            except Exception as e:
                                # Exceptions that were not caught and handled in the code, which lead to failure for the whole chunk
                                group = futures[future]
                                print(f'Exception "{e}" occur for index group: {group}')
                    except KeyboardInterrupt:
                        ex.shutdown(wait=False, cancel_futures=True)
                        raise ValueError('Keyboard interrupt')          
            else:
                # Run on a single core
                filled, fails = self.multi_centroid(to_fill.index)
                self.centroid = pd.concat([self.centroid, filled])
                self.centroid = self.centroid[~self.centroid.index.duplicated(keep='last')]
                
                for fail in fails:
                    print(f'Exception {fail[1]} occur for index: {fail[0]}')
        
        # Save data if specified when class was initialised                           
        if self.save_output == 'full':
            outfile = self.output / f'centroiddata_{self.save_suffix}.csv'
            self.centroid.to_csv(outfile)
        
        # Mark the process as completed         
        self.centroiddata = True
        
        
    def multi_centroid(self, indices):
        """
        Runs the centroid_data generation for a group of target/candidate/sector entries
        
        Parameters:
        indices - An array of target/candidate/sector indices
        """
        results = []
        fails = []
        # Loop through the indices to calculate the centroid offset for the candidate on each sector
        for idx in indices:
            ticid, cndt, sec = idx
            try:
                # Store the results in a list, to be used for the construction of a dataframe
                results.append(self.observed_centroid_offset(ticid, cndt, sec))
            except Exception as e:
                # Store the failure cases and their associated exception
                fails.append((idx, e))
        
        # Construct dataframe from all individual results to return a final datframe for the whole group        
        filled = pd.DataFrame([r for r in results], columns=['ticid', 'candidate', 'sector', 'cam', 'ccd', 'X_diff', 'X_err', 'Y_diff', 'Y_err', 'flag'])
        filled.set_index(['ticid', 'candidate', 'sector'], inplace=True)
        
        return filled, fails
    
                          
    def observed_centroid_offset(self, ticid, cndt, sec):
        """
        Determine per sector centroid offset for a candidate
        
        Parameters:
        ticid - target TIC ID
        cndt - candidate number or toi id
        sec - TESS sector on which the target was observed
        """
        
        # Retrieve the per sector data [per, t0, tdur, tdu23]
        event_data = self.sector_data.loc[(ticid, cndt, sec)]
        tc_per = event_data.per
        tc_t0 = event_data.t0
        tc_tdur = event_data.sec_tdur
        tc_tdur23 = event_data.sec_tdur23

        # Lightcurve directory
        lcdir = self.data.loc[(ticid, cndt)].lcdir

        # Sector lightcurve file
        lc_file = list(lcdir.glob(f'*s{sec:04}*lc.fits'))[0]
        
        # Load in the detrended, normalized and with outliers removed vertical and horizontal centroid position data. 
        # The flag specifies issues found during the loading of the data that prevent the offset calculation.
        # CAM and CCD retrieve for diagnostic purposes when displaying the results.
        time, data_X, data_Y, cent_flag, cam, ccd = utils.load_spoc_centroid(lc_file,
                                                                                flatten=True, cut_outliers=5, trim=True,
                                                                                transitcut=False,
                                                                                tc_per=tc_per,
                                                                                tc_t0=tc_t0,
                                                                                tc_tdur=tc_tdur)
        
        if not cent_flag:
            # Calculate the centroid offsets and their associated errors by fitting a trapezium model
            X_diff, X_diff_err, Y_diff, Y_diff_err, cent_flag = utils.centroid_fitting(ticid, cndt, sec,
                                                                                       time, data_X, data_Y,
                                                                                       tc_per, tc_t0,
                                                                                       tc_tdur, tc_tdur23, 
                                                                                       loss='huber', plot=self.plot_centroid)

            # Handle fitting failure without being flagged
            if  (np.isnan(X_diff) or np.isnan(Y_diff)) and not cent_flag:
                cent_flag = 'Nan from fitting'
        else:
            # Set offset to nan if the sector was not suitable for calculation
            X_diff, X_diff_err, Y_diff, Y_diff_err = np.nan, np.nan, np.nan, np.nan
            
        return (ticid, cndt, sec, cam, ccd, X_diff, X_diff_err, Y_diff, Y_diff_err, cent_flag)
    
  
    def estimate_flux_fractions(self, rerun=False):
        """
        Determine the flux fraction contribution in the TESS aperture of the target stars and their nearby sources
        
        Parameters:
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if flux fractions were previously determined.
        """
        
        # Check if the per sector data generation process has been completed. Raise an error if not.
        if not self.sectordata:
            raise ValueError('Run generate_sector_data first')
        
        for ticid in self.sources.keys():
            # Retrieve the source object data container
            source_obj = self.sources[ticid]

            if source_obj.nearby_fractions is None or rerun:
                # Initialise a dataframe, with the same index as that of the nearby data, to store the fractions
                source_obj.nearby_fractions = pd.DataFrame(index=source_obj.nearby_data.index)
            
            # Retrieve the sectors    
            sectors = self.data.loc[ticid].iloc[0].sectors

            # Check if sector fractions already present in the dataframe.
            # Difference will be only if the nearby_fractions dataframe was already existing in the source_obj data.
            # Allows to run the process only for new sectors
            already_run = np.array([int(x[1:]) for x in source_obj.nearby_fractions.columns])
            sectors = np.setdiff1d(np.array(sectors), already_run)
                                     
            for sec in sectors:
                # Retrieve the lc file for the sector
                file_loc = self.data.loc[ticid].iloc[0].lcdir
                filepath = list(file_loc.glob(f'*s{sec:04}*lc.fits'))[0]
                
                # Load in the pipeline aperture and centroid masks for the target pixels, the wcs and the origin location of the target pixel on the ccd
                aperture_mask, centroid_mask, wcs, origin, cam, ccd = utils.load_spoc_masks(filepath)
                # Store the data in the object, so that they can be reused
                source_obj.wcs[sec] = wcs
                source_obj.origin[sec] = origin
                source_obj.aperture_mask[sec] = aperture_mask
                source_obj.centroid_mask[sec] = centroid_mask
                source_obj.scc.loc[sec] = [cam, ccd]
                
                # Create temporary copy of the nearby data
                temp_data = source_obj.nearby_data.copy()
                
                # Compare the estimated and observed flux of the target in the sector to correct for discrepancies 
                estimated_flux = temp_data.loc[ticid, 'Flux']
                dummy_cndt = self.data.loc[ticid].index[0]
                observed_flux = self.sector_data.loc[(ticid, dummy_cndt, sec), 'sec_flux']
                cor_factor = observed_flux/estimated_flux
                temp_data['Flux'] *= cor_factor
                
                # Store the correction factor for later use
                source_obj.cor_factor[sec] = cor_factor
                
                # Account for proper motion
                obs_time = self.sector_data.loc[(ticid, dummy_cndt, sec), 'sec_time']
                cor_ra, cor_dec = utils.gaia_pm_corr(obs_time, temp_data['ra'].values, temp_data['dec'].values, temp_data['pmra'].values, temp_data['pmdec'].values)
                temp_data['ra'] = cor_ra
                temp_data['dec'] = cor_dec
                
                # Use the retrieved WCS to convert the ra and dec of the nearby sources into pixel locations
                temp_data['x'], temp_data['y'] = wcs.all_world2pix(temp_data.ra, temp_data.dec, 0)
                
                # Test if target position from wcs is correctly in the aperture. Catch wcs errors
                in_ap = utils.test_target_aperture(temp_data.loc[ticid].x, temp_data.loc[ticid].y, aperture_mask)
                
                if in_ap:
                    # Calculate the flux fractions and the total flux in aperture, by modelling the observation using the TESS PRF
                    prfs = utils.sources_prf(sec, cam, ccd, origin, temp_data['x'].values, temp_data['y'].values, aperture_mask.shape)
                    fractions, total_flux = utils.flux_fraction_in_ap(temp_data.Flux.values, aperture_mask, prfs)
                else:
                    # Set the fractions to nan, to effectively ignore this sector
                    prfs = None
                    fractions = np.zeros(len(temp_data))
                    fractions[:] = np.nan
                    
                    total_flux = np.nan
                
                # Store the model prfs, fractions, the modeled total flux and the Tmag equivalent of the flux in the aperture
                source_obj.prfs[sec] = prfs
                source_obj.nearby_fractions[f'S{sec}'] = fractions 
                source_obj.totalflux[sec] = total_flux
                source_obj.totalmag_equivalent[sec] = 10 - 2.5*np.log10(total_flux/15000)
                     
        # Save output if specified when class was initialised
        if self.save_output == 'full':
            outfile = self.output / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
        
        # Mark the process as completed       
        self.flux_fractions = True


    def estimate_nearby_depths(self, rerun=False):
        """
        Determine the flux fraction contribution in the TESS aperture of the target stars and their nearby sources
        
        Parameters:
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if nearby depths were previously determined.
        """
         # Check if flux fractions have been computed. Raise an error if not.
        if not self.flux_fractions:
            raise ValueError('Run estimate_flux_fractions first')
        
        for ticid in self.sources.keys():
            # Retrieve the target data and source object
            target_data = self.data.loc[ticid]
            source_obj = self.sources[ticid]
            
            candidates = target_data.index.values
            
            # Construct a dataframe to store the depth of the event if it would occur in each of the nearby sources,
            # based on the depth of the event detected on the target on each sector. Initialised with nan.
            indx = pd.MultiIndex.from_product([source_obj.nearby_data.index.values, candidates], names=['ticid', 'candidate'])
            cols = [f'S{sec}' for sec in target_data.iloc[0].sectors]
            nearby_depths = pd.DataFrame(data=np.nan, index=indx, columns=cols)
            
            if not rerun:
                # Merged the new dataframe with the existing on, dropping the Mean depth column
                try:
                    nearby_depths = pd.concat([nearby_depths, source_obj.nearby_depths.drop('Mean', axis=1)])
                    nearby_depths = nearby_depths[~nearby_depths.index.duplicated(keep='last')]
                except KeyError:
                    pass
            
            # Store the new dataframe on the object
            source_obj.nearby_depths = nearby_depths

            # Fill in the depths per candidate
            for cndt in candidates:
                # Retrieve the existing candidate depth data
                sub_depths = source_obj.nearby_depths.query(f'candidate == {cndt}')
                # Identify the sector columns which are still not filled, i.e. all entries are null
                null_sectors = sub_depths.loc[:, sub_depths.isnull().all()].columns

                # Per sector
                for sec in null_sectors:
                    # Convert the 'S{}' column name to int
                    sec_num = int(sec[1:])

                    # Retrieve the candidate depth on target from the per-sector data
                    cndt_sec_depth = self.sector_data.loc[(ticid, cndt, sec_num), 'sec_depth']
                    
                    # If the depth was less than 50ppm, set it to nan, to effectively skip the sector 
                    if cndt_sec_depth < 50:
                        cndt_sec_depth = np.nan

                    # Retrieve the target and nearby flux fractions
                    f_target = source_obj.nearby_fractions.loc[ticid, sec]
                    f_nearby = source_obj.nearby_fractions[sec].values
                    
                    # Calculate the implied depths for the nearby sources and store them
                    depths = utils.nearby_depth(cndt_sec_depth, f_target, f_nearby)
                    source_obj.nearby_depths.loc[source_obj.nearby_depths.index.get_level_values('candidate') == cndt, sec] = depths
            
            # Calculate the mean eclipse depth for the event based on all sectors, ignoring sectors with 0 depth
            source_obj.nearby_depths['Mean'] = source_obj.nearby_depths.replace(0, np.nan).mean(axis=1)

            # For nearby sources where the flux fraction in aperture was 0 for all sectors, set the mean to 0.
            zero_idx = source_obj.nearby_fractions[(source_obj.nearby_fractions == 0).all(axis=1)].index
            source_obj.nearby_depths.loc[zero_idx, 'Mean'] = 0
        
        # Save output if specified when class was initialised
        if self.save_output == 'full':
            outfile = self.output / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)   
        
        # Mark the process as completed   
        self.estimate_depths = True
    
              
    def generate_probabilities(self, max_eclipsedepth=1.0, prob_thresh=1e-4, rerun=False):
        """
        Generates positional probabilities for the target and nearby sources and performs an assessment of the suitability for each to be the true host of the event.
        
        Parameters:
        max_eclipsedepth - The maximum depth allowed for an eclipse on a source to be considered valid
        max_transitdepth = The maximum depth allowed for a transit on a source to be considered as a possible planet candidate
        prob_thresh - The minimum probability for a source to be considered as a possible alternative source of the detected event
        rerun - force the process to rerun, irrespective of whether a load_suffix was provided during the class initialization or if probabilities were previously generated.
        """
        # Check if centroid data have been generated. Raise an error if not.
        if not self.centroiddata:
            raise ValueError('Run generate_centroiddata first!')
        
        # Check if nearby depths have been determined. Raise an error if not.
        if not self.estimate_depths:
            raise ValueError('Run nearby_flux_fractions first!')
        
        print(f'Generating Positional Probabilities')
        
        for targetid in self.sources.keys():
            # Retrieve the target data and source object
            target_data = self.data.loc[targetid]
            source_obj = self.sources[targetid]

            sectors = target_data.iloc[0].sectors
            
            if not rerun:
                # Run only for sectors not processed yet
                sectors_out = np.setdiff1d(sectors, source_obj.cent_out.index)
            else:
                sectors_out = sectors
            
            for sec in sectors_out:
                # Retrieve the model prfs stored before
                prfs = source_obj.prfs[sec]
                if prfs is not None:
                    temp_data = source_obj.nearby_data[['Flux']].copy()
                    cor_factor = source_obj.cor_factor[sec]
                    temp_data['Flux'] *= cor_factor
                    
                    centroid_mask = source_obj.centroid_mask[sec]
                    
                    cent_x, cent_y = utils.model_centroid(centroid_mask, temp_data.Flux.values, prfs)
                else:
                    cent_x, cent_y = np.nan, np.nan
                                    
                # Store the out of transit centroid for each sector
                source_obj.cent_out.loc[sec] = cent_x, cent_y

            # Construct a dataframe to store the model centroid in transit per sector for the event on the target and on the nearby sources
            cent_in = pd.DataFrame(data=0.0, 
                                   index=pd.MultiIndex.from_product([source_obj.nearby_data.index, target_data.index, sectors], names=['ticid', 'candidate', 'sector']), 
                                   columns=['X', 'Y'])
            if not rerun:
                # Merged with existing in transit model centroids
                cent_in = pd.concat([cent_in, source_obj.cent_in])
                cent_in = cent_in[~cent_in.index.duplicated(keep='last')]
                
            source_obj.cent_in = cent_in
            
            # Determine which entries still need to be processed                                                
            to_fill = source_obj.cent_in.loc[(cent_in == 0).all(axis=1)]
            if len(to_fill) > 0:
                for sec in to_fill.index.unique('sector'):
                    # Check if out of transit centroid for the sector is nan
                    if source_obj.cent_out.loc[sec].isna().any():
                        source_obj.cent_in.loc[cent_in.query(f'sector == {sec}').index] = np.nan
                        continue
                        
                    # Retrieve the sector information
                    temp_data = source_obj.nearby_data[['Flux']].copy()
                    cor_factor = source_obj.cor_factor[sec]
                    temp_data['Flux'] *= cor_factor
                    centroid_mask = source_obj.centroid_mask[sec]
                    prfs = source_obj.prfs[sec]
                    
                    # Process each candidate individually
                    for cndt in to_fill.query(f'sector == {sec}').index.unique('candidate'):
                        # Check if the observed centroid for the candidate is nan. No need to model centroid then.
                        if self.centroid.loc[targetid, cndt, sec].isna().any():
                            source_obj.cent_in.loc[cent_in.query(f'candidate == {cndt} & sector == {sec}').index] = np.nan
                            continue
                                                                    
                        # Check if the candidate depth for the sector is nan
                        cndt_depth = source_obj.nearby_depths.loc[(targetid, cndt), f'S{sec}']
                        if np.isnan(cndt_depth):
                            source_obj.cent_in.loc[cent_in.query(f'candidate == {cndt} & sector == {sec}').index] = np.nan
                            continue
                        
                        # Determine the model in transit centroid for the target and the nearby sources
                        for ticid in source_obj.nearby_data.index.unique('ticid'):
                            # Retrieve the mean depth for the source
                            depth = source_obj.nearby_depths.loc[(ticid, cndt), f'Mean']*1e-6
                            obj_type = source_obj.nearby_data.loc[ticid, 'objType']

                            # Check depth suitability
                            if depth == 0.0 or np.isnan(depth):
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            elif depth*0.9 > max_eclipsedepth:
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            elif obj_type != 'STAR':
                                source_obj.cent_in.loc[ticid, cndt, sec] = np.nan
                            else:
                                # Retrieve the fluxes
                                fluxes = temp_data['Flux'].copy()
                                
                                # Scale the flux of the source by the depth
                                depth_scale = np.min((depth, 1.0))
                                fluxes.loc[ticid] *= (1-depth_scale) 
                                
                                # Determine in transit model centroid
                                X, Y = utils.model_centroid(centroid_mask, fluxes.values, prfs)
                                
                                # Store the model centroids
                                source_obj.cent_in.loc[ticid, cndt, sec] = [X, Y]

                # Determine the model centroid offset and the error for all candidates and sources
                model_offset = pd.DataFrame()
                model_offset['X_diff'] = source_obj.cent_in['X'] - source_obj.cent_out['X']
                model_offset['Y_diff'] = source_obj.cent_in['Y'] - source_obj.cent_out['Y']
                model_offset['X_err'] = model_offset['X_diff']*0.1
                model_offset['Y_err'] = model_offset['Y_diff']*0.1
                
                # Store the model centroid offset and errors                
                source_obj.model_centroid = model_offset[['X_diff', 'X_err', 'Y_diff', 'Y_err']]
                
                # Add the observed centroid offset to the dataframe, per sector, to ease subsequent calculations
                model_offset = model_offset.join(self.centroid.loc[targetid], on=['candidate', 'sector'], rsuffix='_obs')
                
                # Probabilistically compare the observed and model centroid offsets
                model_offset['Probability'] = model_offset.apply(lambda x: utils.calc_centroid_probability(x['X_diff_obs'], x['Y_diff_obs'], 
                                                                                                            x['X_err_obs'], x['Y_err_obs'], 
                                                                                                            x['X_diff'], x['Y_diff'], 
                                                                                                            x['X_err'], x['Y_err']), axis=1)
                
                # Compute the sum of the probabilities for all candidates per sector
                model_offset = model_offset.join(model_offset.groupby(['candidate', 'sector']).agg(Prob_Sum=('Probability', 'sum')), on=['candidate', 'sector'])
                                
                # Compute the normalised probability for each candidate per sector
                model_offset['Norm_Probability'] = model_offset['Probability'] / model_offset['Prob_Sum']
                
                # Store the probability, probability sum and normalised probability per sector/candidate/source 
                source_obj.model_prob = model_offset[['Probability', 'Prob_Sum', 'Norm_Probability']]
                
                # Construct a dataframe for the probababilities to be reported
                prob_centroid = pd.DataFrame(index=pd.MultiIndex.from_product([source_obj.nearby_data.index, target_data.index], names=['ticid', 'candidate']))
                
                # Calculate the Max, Mean and Median un-normalised probability for each source
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MaxProb=('Probability','max')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MeanProb=('Probability','mean')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MedianProb=('Probability','median')), on=['ticid', 'candidate'])
                
                # Calculate the Max, Mean and Median normalised probability for each source
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MaxNormProb=('Norm_Probability','max')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MeanNormProb=('Norm_Probability','mean')), on=['ticid', 'candidate'])
                prob_centroid = prob_centroid.join(model_offset.groupby(['ticid', 'candidate']).agg(MedianNormProb=('Norm_Probability','median')), on=['ticid', 'candidate'])
                
                # Calculate the sum for the median normalised probability
                prob_centroid = prob_centroid.join(prob_centroid.groupby(['candidate']).agg(NormMedianSum=('MedianNormProb','sum')), on=['candidate'])

                # Calculate the final positional probability
                prob_centroid['PositionalProb'] = prob_centroid['MedianNormProb'] / prob_centroid['NormMedianSum']
                
                # Drop the probability sum
                prob_centroid.drop(['NormMedianSum'], axis=1, inplace=True)
                                            
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
                nearby_assessment.loc[source_obj.prob_centroid[mask].query(f'PositionalProb < {prob_thresh}').index] = False, 'Centroid probability below threshold'
                # Nan
                nearby_assessment.loc[source_obj.prob_centroid[mask].query(f'PositionalProb != PositionalProb').index] = False, 'Nan centroid probability'
                                
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
            
            nb_assessment = source_obj.nearby_assessment.copy()
            nb_assessment['target'] = targetid
            nb_assessment.reset_index(inplace=True)
            nb_assessment.rename(columns={'ticid':'source'}, inplace=True)
            nb_assessment.set_index(['target', 'candidate', 'source'], inplace=True)
            
            self.assessment = pd.concat([self.assessment, nb_assessment])
        
        self.probabilities = self.probabilities[~self.probabilities.index.duplicated(keep='last')]    
        self.probabilities.sort_values(['target', 'candidate', 'PositionalProb'], ascending=[1,1,0], inplace=True)
        
        self.assessment = self.assessment[~self.assessment.index.duplicated(keep='last')]  
        self.assessment.sort_values('Possible', ascending=False)
        
        print('Positional Probabilities generated!')
        
        self.probabilities.rename(columns={'PositionalProb':'PositionalProbability',
                                           'disp':'Disposition'}, inplace=True)
                 
        if self.save_output == 'full':  
            # Output the probabilities
            outfile = self.output / f'Probabilities_{self.save_suffix}.csv'
            self.probabilities.to_csv(outfile)
            
            # Output the assessment
            outfile = self.output / f'Assessment_{self.save_suffix}.csv'
            self.assessment.to_csv(outfile)
            
            # Save the sources to be reused
            outfile = self.output / f'sources_{self.save_suffix}.pkl'
            with open(outfile, 'wb') as f:
                pickle.dump(self.sources, f, protocol=5)
        elif self.save_output is True:
            # Output just the Positional Probabilities
            outfile = self.output / f'Probabilities_{self.save_suffix}.csv'
            self.probabilities[['PositionalProbability', 'Disposition']].to_csv(outfile)
            

class Source(object):

    def __init__(self, tic, coords=(),  pm=(), mags={}, star_rad=None, star_mass=None, teff=None):
        """
        Datastore for a flux source
        """
        self.TIC = tic # Target TIC ID
        self.coords = coords # Target ra and dec
        self.pm = pm # Target proper motion
        self.mags = mags # Target TESS, V and GAIA magnitudes
        self.star_rad = star_rad # Target Stellar Radius
        self.star_mass = star_mass # Target Stellar Mass
        self.teff = teff # Target effective temperature
        self.scc = pd.DataFrame(columns=['sector', 'cam', 'ccd']).set_index('sector') # TESS sector/cam/ccd on which the target was observed
        self.wcs = {} # Per sector WCS 
        self.origin = {} # Per sector target pixel origin location
        self.aperture_mask = {} # Per sector pixels used for the aperture
        self.centroid_mask = {} # Per sector pixels used to determine the photometric centroid
        self.nearby_data = None # Data for the target's nearby sources (includes the target)
        self.cor_factor = {} # Correction fraction to match model and observed target flux
        self.nearby_fractions = None # Flux fraction contribution of nearby stars in aperture
        self.nearby_depths = pd.DataFrame() # Implied eclipse depth for target's nearby sources
        self.totalflux = {} # Model total flux in aperture
        self.totalmag_equivalent = {} # Model magnitude equivalent of total flux in aperture
        self.prfs = {} # Sector Pixel-Response Function
        self.cent_out = pd.DataFrame(columns=['sector', 'X', 'Y']).set_index('sector') # Model centroid out of transit
        self.cent_in = pd.DataFrame() # Model centroid in-transit
        self.model_centroid = pd.DataFrame() # Model centroid offset
        self.prob_centroid = pd.DataFrame() # Positional Probability
        self.nearby_assessment = None # Assessment of nearby targets
