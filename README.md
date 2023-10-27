# TESSPositionalProbability

A method for deriving probabilistic estimates for the true sources of TESS transiting events. 

Method Overview: https://arxiv.org/abs/2310.15833

## Requirements:
python >= 3.10
pandas  >= 1.4.4
numpy
scipy
astropy
astroquery
matplotlib

## Installation:
- Clone or download the repository.
- Set up a python environment based on the requirements. 
- The repository includes an example input file and data. Run:

  ```python PositionalProbability.py```

  to test that method has been setup and working correctly.

## Usage:
- Create an input csv file or modify the included example.csv in the input folder for your data. Add the following:
  - TIC ID of target star
  - Candidate Identification (usually TOI number)
  - Period (days)
  - transit epoch (BTJD)
  - transit duration (days)
  - transit depth (ppm)
- Add the SPOC lightcurve files for your targets in folders with their respective TIC ID number inside the Lightcurve folder, similar to the provided example data.
- Modify the PositionalProbability.py script, replacing 'example.csv' with your input filename.
- Set the 'save_output' parameter to True to output the probabilities in a csv file or False to print the probabilities in the command line. Provide a suitable save_suffix if saving the output.
- Run PositionalProbability.py

A detailed explanation of the user functions and their parameters can be found in CandidateSet.py inside the CandidateSet folder. 
