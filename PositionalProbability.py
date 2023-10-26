from CandidateSet import CandidateSet as cs

# Specify input, output and execution parameters. Load input data and initialize.
cset = cs.CandidateSet('example.csv', save_output=True, save_suffix='_Example', multiprocessing=0)

# Retrieve required data for the target stars from the TIC, including the nearby sources.
cset.generate_sources(infile=None)
if not cset.find_stars:
    # Connection errors can lead to some sources not being created. Running the process again should resolve this
    print('Some sources failed to be created. Attempting to retrieve them again...')
    cset.generate_sources()
    cset.find_stars = True

# Determine the per sector transit characteristics
cset.generate_per_sector_data(infile=None)

# Measure the observed photometric centroid
cset.generate_centroiddata(infile=None)

# Produce estimated flux contributions of nearby sources in the aperture
cset.estimate_flux_fractions()

# Determine the implied eclipse depth
cset.estimate_nearby_depths()

# Generate positional probabilities
cset.generate_probabilities()

