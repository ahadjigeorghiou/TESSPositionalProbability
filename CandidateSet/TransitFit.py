import numpy as np
from scipy import optimize
from CandidateSet import utils

def Trapezoidmodel(phase_data, t0_phase, t23, t14, depth):
    centrediffs = np.abs(phase_data - t0_phase)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    if t23>t14:
        model = np.ones(len(phase_data))*1e8
    return model
    
def Trapezoidmodel_fixephem(phase_data, t23, t14, depth):
    centrediffs = np.abs(phase_data - 0.5)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    if t23>t14:
        model = np.ones(len(phase_data))*1e8
    return model


class TransitFit(object):

    def __init__(self,lc,initialguess,fixper=None,fixt0=None):
        self.lc = lc
        self.init = initialguess
        self.fixper = fixper
        self.fixt0 = fixt0        

        self.params,self.cov = self.FitTrapezoid()
        if not np.isnan(self.cov).all():
            self.errors = np.sqrt(np.diag(self.cov))
        else:
            self.errors = np.full_like(self.params, -10)
            
        if self.fixt0 is None:  #then convert t0 back to time
            self.params[0] = (self.params[0]-0.5)*self.fixper + self.initial_t0
            self.errors[0] *= self.fixper 
    
    def FitTrapezoid(self):
        if self.fixt0 is None:
            self.initial_t0 = self.init[0]
            phase = utils.phasefold(self.lc['time'],self.fixper,
            					    self.initial_t0+self.fixper/2.)  #transit at phase 0.5
            initialguess = self.init.copy()
            initialguess[0] = 0.5
            bounds =  [(0.45, 0.0001, 0.0001, initialguess[3]*0.99),(0.55, 0.35, 0.35, 1)]    
            try:
                fit = optimize.curve_fit(Trapezoidmodel, phase, self.lc['flux'], 
                                        p0=initialguess, sigma=self.lc['error'],
                                        bounds=bounds, absolute_sigma=False)
            except (RuntimeError, ValueError):
                nan_arr = np.array([np.nan, np.nan, np.nan, np.nan])
                fit = (nan_arr, np.nan)
        else:
            phase = utils.phasefold(self.lc['time'],self.fixper,
            				        self.fixt0+self.fixper/2.)  #transit at phase 0.5
            initialguess = self.init.copy()
            bounds =  [(0.0001, 0.0001, 0),(0.35, 0.35, 1)]
            try:
                fit = optimize.curve_fit(Trapezoidmodel_fixephem, phase, self.lc['flux'], 
                                        p0=initialguess, sigma=self.lc['error'],
                                        bounds=bounds, absolute_sigma=False)
            except (RuntimeError, ValueError):
                nan_arr = np.array([np.nan, np.nan, np.nan])
                fit = (nan_arr, np.nan)
        return fit[0], fit[1]