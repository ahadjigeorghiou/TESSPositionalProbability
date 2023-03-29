import numpy as np
from scipy import optimize
import batman
from . import utils

def Trapezoidmodel(phase_data, t0_phase, t23, t14, depth):
    centrediffs = np.abs(phase_data - t0_phase)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    if t23>t14:
        model = np.ones(len(phase_data))*1e8
    return model
    
def Trapezoidfitfunc(fitparams,y_data,y_err,x_phase_data):
    t0 = fitparams[0]
    t23 = fitparams[1]
    t14 = fitparams[2]
    depth = fitparams[3]
        
    # if (t0<0.2) or (t0>0.8) or (t23 < 0) or (t14 < 0) or (t14 < t23) or (depth < 0):
    #     return np.ones(len(x_phase_data))*1e8
    # if (np.abs(t14-init_tdur)/init_tdur) > 0.05:
    #     return np.ones(len(x_phase_data))*1e8
    if (t14 < t23):
        return np.ones(len(x_phase_data))*1e8
    
    model = Trapezoidmodel(t0,t23,t14,depth,x_phase_data)
    return (y_data - model)/y_err

def Trapezoidmodel_fixephem(phase_data, t23, t14, depth):
    centrediffs = np.abs(phase_data - 0.5)
    model = np.ones(len(phase_data))
    model[centrediffs<t23/2.] = 1-depth
    in_gress = (centrediffs>=t23/2.)&(centrediffs<t14/2.)   
    model[in_gress] = (1-depth) + (centrediffs[in_gress]-t23/2.)/(t14/2.-t23/2.)*depth
    if t23>t14:
        model = np.ones(len(phase_data))*1e8
    return model

def Transitfitfunc(fitparams,y_data,y_err,x_data,m,bparams,init_per,init_t0):
    per = fitparams[0]
    t0 = fitparams[1]
    arstar = fitparams[2]
    rprstar = fitparams[3]
#    inc = fitparams[4]
#    inc_rad = inc*np.pi/180. 
#    if (rprstar < 0) or (arstar < 1.5) or (np.cos(inc_rad) > 1./arstar) or ((t0-init_t0)/init_t0 > 0.05) or ((per-init_per)/init_per > 0.05):
    if (rprstar < 0) or (arstar < 1.5) or ((t0-init_t0)/init_t0 > 0.05) or ((per-init_per)/init_per > 0.05):
        return np.ones(len(x_data))*1e8
    bparams.t0 = t0                        #time of inferior conjunction
    bparams.per = per                      #orbital period
    bparams.rp = rprstar                   #planet radius (in units of stellar radii)
    bparams.a = arstar                     #semi-major axis (in units of stellar radii)
    flux = m.light_curve(bparams)
    return (y_data - flux)/y_err

def Transitfitfunc_fixephem(fitparams,y_data,y_err,x_data,m,bparams):
    arstar = fitparams[0]
    rprstar = fitparams[1]
#    inc = fitparams[4]
#    inc_rad = inc*np.pi/180. 
#    if (rprstar < 0) or (arstar < 1.5) or (np.cos(inc_rad) > 1./arstar) or ((t0-init_t0)/init_t0 > 0.05) or ((per-init_per)/init_per > 0.05):
    if (rprstar < 0) or (arstar < 1.5):
        return np.ones(len(x_data))*1e8
    bparams.rp = rprstar                      #planet radius (in units of stellar radii)
    bparams.a = arstar                        #semi-major axis (in units of stellar radii)
    flux = m.light_curve(bparams)
    return (y_data - flux)/y_err


class TransitFit(object):

    def __init__(self,lc,initialguess,exp_time,sfactor,fittype='model',
    			 fixper=None,fixt0=None):
        self.lc = lc
        self.init = initialguess
        self.exp_time = exp_time
        self.sfactor = sfactor
        self.fixper = fixper
        self.fixt0 = fixt0        
        if fittype == 'model':
            self.params,self.cov = self.FitTransitModel()
            self.errors,self.chisq = self.GetErrors()
        elif fittype == 'trap':
            self.params,self.cov = self.FitTrapezoid()
            print(np.sqrt(np.diag(self.cov))[-1]*1e6)
            if self.fixt0 is None:  #then convert t0 back to time
                self.params[0] = (self.params[0]-0.5)*self.fixper + self.initial_t0   
    
    def FitTrapezoid(self):
        if self.fixt0 is None:
            self.initial_t0 = self.init[0]
            phase = utils.phasefold(self.lc['time'],self.fixper,
            					    self.initial_t0+self.fixper/2.)  #transit at phase 0.5
            initialguess = self.init.copy()
            initialguess[0] = 0.5
            bounds =  [(0.45, initialguess[2]*0.01, initialguess[2]*0.9, 0),(0.55, initialguess[2]*1.1, initialguess[2]*1.1, 1)]    
            # fit = optimize.leastsq(Trapezoidfitfunc, initialguess, 
            # 					   args=(self.lc['flux'],self.lc['error'],phase, initialguess[2]),
            # 					   full_output=True)
            fit = optimize.curve_fit(Trapezoidmodel, phase, self.lc['flux'], 
                                     p0=initialguess, sigma=self.lc['error'],
                                     bounds=bounds, absolute_sigma=False)
        else:
            self.initial_t0 = self.fixt0
            phase = utils.phasefold(self.lc['time'],self.fixper,
            				        self.initial_t0+self.fixper/2.)  #transit at phase 0.5
            initialguess = self.init.copy()
            initialguess = initialguess[1:]
            bounds =  [(initialguess[1]*0.001, initialguess[1]*0.8, 0),(initialguess[1]*1.2, initialguess[1]*1.2, 1)]
            try:
                fit = optimize.curve_fit(Trapezoidmodel_fixephem, phase, self.lc['flux'], 
                                        p0=initialguess, sigma=self.lc['error'],
                                        bounds=bounds, absolute_sigma=False)
            except RuntimeError:
                nan_arr = np.array([np.nan, np.nan, np.nan])
                fit = (nan_arr, nan_arr)
        return fit[0], fit[1]

    def FitTransitModel(self):
        #initialguess = np.array([init_per,init_t0,init_arstar,init_rprstar,init_inc])
        fix_e = 0.
        fix_w = 90.
        ldlaw = 'quadratic'
        fix_ld = [0.1,0.3]
        bparams = batman.TransitParams()    #object to store transit parameters
        bparams.rp = self.init[3]           #planet radius (in units of stellar radii)
        bparams.a = self.init[2]            #semi-major axis (in units of stellar radii)
        bparams.inc = 90.
        bparams.ecc = fix_e                 #eccentricity
        bparams.w = fix_w                   #longitude of periastron (in degrees)
        bparams.limb_dark = ldlaw           #limb darkening model
        bparams.u = fix_ld                  #limb darkening coefficients
        if self.fixper is None:
            bparams.t0 = self.init[1]       #time of inferior conjunction
            bparams.per = self.init[0]      #orbital period
            m = batman.TransitModel(bparams, self.lc['time'],exp_time=self.exp_time,
            						supersample_factor=self.sfactor)
            fit = optimize.least_squares(Transitfitfunc, self.init.copy().flatten(), 
            					        args=(self.lc['flux'],self.lc['error'],self.lc['time'],
            					        m,bparams,self.init[0],self.init[1]),
                                        bounds=[(0.9999999*self.init[0], self.init[1]-0.5, 1.5, 0), (1.0000000*self.init[0], self.init[1]+0.5, np.inf, np.inf)])
        else:
            bparams.t0 = self.fixt0
            bparams.per = self.fixper
            m = batman.TransitModel(bparams, self.lc['time'],exp_time=self.exp_time,
            						supersample_factor=self.sfactor)    
            fit = optimize.leastsq(Transitfitfunc_fixephem, self.init.copy()[1:], 
            					   args=(self.lc['flux'],self.lc['error'],self.lc['time'],
            					   m,bparams),full_output=True)            
        return fit['x'],None   

    def GetErrors(self):
        bparams = batman.TransitParams()    #object to store transit parameters
        bparams.inc = 90.
        bparams.ecc = 0.                    #eccentricity
        bparams.w = 90.                     #longitude of periastron (in degrees)
        bparams.limb_dark = 'quadratic'     #limb darkening model
        bparams.u = [0.1,0.3]               #limb darkening coefficients
        if self.fixper is None:
            bparams.t0 = self.params[1]     #time of inferior conjunction
            bparams.per = self.params[0]    #orbital period
            bparams.rp = self.params[3]     #planet radius (in units of stellar radii)
            bparams.a = self.params[2]      #semi-major axis (in units of stellar radii)

        else:
            bparams.t0 = self.fixt0
            bparams.per = self.fixper
            bparams.rp = self.params[1]     #planet radius (in units of stellar radii)
            bparams.a = self.params[0]      #semi-major axis (in units of stellar radii)

        m = batman.TransitModel(bparams, self.lc['time'],exp_time=self.exp_time,
        						supersample_factor=self.sfactor)          
        flux = m.light_curve(bparams)
        if self.fixper is None:
            if self.cov is not None:
                s_sq = np.sum(np.power((self.lc['flux'] - flux),2))/(len(self.lc['flux'])-4)
                err = (np.diag(self.cov*s_sq))**0.5
            else:
                #print('Fit did not give covariance, error based features will not be meaningful')
                err = np.ones(4)*-10
            chisq = 1./len(self.lc['flux']-4) * np.sum(np.power((self.lc['flux'] - flux)/self.lc['error'],2))
        else:
            if self.cov is not None:
                s_sq = np.sum(np.power((self.lc['flux'] - flux),2))/(len(self.lc['flux'])-2)
                err = (np.diag(self.cov*s_sq))**0.5
            else:
                #print('Fit did not give covariance, error based features will not be meaningful')
                err = np.ones(2)*-10
            chisq = 1./len(self.lc['flux']-2) * np.sum(np.power((self.lc['flux'] - flux)/self.lc['error'],2))   
        return err,chisq
