from CandidateSet import CandidateSet as cs

cset = cs.CandidateSet('example.csv', save_output=True)

cset.find_TIC_stars()
if not cset.find_stars:
    # Connection errors can lead to some sources not being created. Running the process again should resolve this
    print('Some sources failed to be created. Attempting to retrieve them again...')
    cset.find_TIC_stars()
    cset.find_stars = True

cset.generate_per_sector_data()

cset.generate_centroiddata()

cset.estimate_flux_fractions()

cset.estimate_nearby_depths()

cset.generate_probabilities()

