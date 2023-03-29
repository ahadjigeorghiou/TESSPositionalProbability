from CandidateSet import CandidateSet as cs

cset = cs.CandidateSet('example.csv')

cset.find_TIC_stars()

cset.generate_per_sector_data()

cset.generate_centroiddata()

cset.estimate_flux_fractions()

cset.estimate_nearby_depths()

cset.cut_faintstars()

