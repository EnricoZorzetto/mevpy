
''' 
###############################################################################

        Enrico Zorzetto, 9/10/2017
        enrico.zorzetto@duke.edu
        
        Set of functions to  calibrate and validate the MEV distribution
        most functions are to be applied to Pandas data frames
        with the following fields:
            'PRCP' :: for the daily rainfall values
            'YEAR' :: for the observation year (in format yyyy)
            'DATE' :: date in format yyyymmdd

###############################################################################
'''

import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import mevpy.gev_fun as gev
from scipy.special import gamma
from scipy.stats import exponweib, pearsonr


###############################################################################
###############################################################################

########################### WEIBULL DISTRIBUTION ##############################

###############################################################################
###############################################################################

def wei_fit(sample, how = 'pwm', threshold = 0, std = False, std_how = 'boot', std_num = 1000):
    ''' fit Weibull with one of the following methods:
        -----------------------------------------------------------------------
        how = 'pwm' for probability weighted moments
        how = 'ml' for maximum likelihood
        how = 'ls' for least squares 
        -----------------------------------------------------------------------
        choice of threshold available for PWM only
        without renormalization (default is zero threshold)
        -----------------------------------------------------------------------
        optional: if std = True (default is false)
        compute parameter est. standard deviations. parstd
        and their covariance matrix varcov
        if std_how = 'boot' bootstrap is used 
        if std_how = 'hess' hessian is used (only available for max like.)
        std_num --> number or resamplings in the bootstrap procedure.
        default is 1000. 
        --------------------------------------------------------------------'''
    # print('how = ', how)
    if   how == 'pwm':
        N, C, W = wei_fit_pwm(sample, threshold = threshold) 
    elif how == 'ml':
        N, C, W = wei_fit_ml(sample)
    elif how == 'ls':
        N, C, W = wei_fit_ls(sample)
    else:
        print(' ERROR - insert a valid fitting method ')
    parhat = N,C,W
    if std == True:
        if how == 'pwm':
            parstd, varcov = wei_boot(sample, fitfun = wei_fit_pwm, npar= 2, ntimes = std_num)
        elif how == 'ls':
            parstd, varcov = wei_boot(sample, fitfun = wei_fit_ls, npar= 2, ntimes = std_num)
        elif how == 'ml' and std_how == 'boot':
            parstd, varcov = wei_boot(sample, fitfun = wei_fit_ml, npar = 2, ntimes = std_num)
        elif how == 'ml' and std_how == 'hess':
            print(" wei_fit ERROR: 'hess' CIs not available yet")
            ni, ci, wi, parstd, varcov = wei_fit_ml(sample, std = True)
        else:
            print('wei_fit ERROR: insert a valid method for CIs')
        return parhat, parstd, varcov
    else:
        return parhat
    
    
def wei_boot(sample, fitfun, npar = 2, ntimes = 1000):
    '''non parametric bootstrap technique 
    for computing confidence interval for a distribution
    (when I do not know the asymptotic properties of the distr.)
    return std and optional pdf of fitted parameters  
    and their covariance matrix varcov
    fit to a sample of a distribution using the fitting function fitfun
    with a number of parameters npar 
    ONLY FOR WEIBULL
    Ignore the first output parameter - N'''
    n = np.size(sample)
    # resample from the data with replacement
    parhats = np.zeros((ntimes,npar))
    for ii in range(ntimes):
        replaced = np.random.choice(sample,n)
        NCW = fitfun(replaced)  
        parhats[ii,:] = NCW[1:]   
    parstd = np.std(parhats, axis = 0)
    varcov = np.cov(parhats, rowvar = False)
    return parstd, varcov    


def wei_fit_pwm(sample, threshold = 0): 
    ''' fit a 2-parameters Weibull distribution to a sample 
    by means of Probability Weighted Moments (PWM) matching (Greenwood 1979)
    using only observations larger than a value 'threshold' are used for the fit
    -- threshold without renormalization -- it assumes the values below are 
    not present. Default threshold = 0    
    INPUT:: sample (array with observations)
           threshold (default is = 0)
    OUTPUT::
    returns dimension of the sample (n) (only values above threshold)
    Weibull scale (c) and shape (w) parameters '''    
    sample = np.asarray(sample) # from list to Numpy array
    wets   = sample[sample > threshold]
    x      = np.sort(wets) # sort ascend by default
    M0hat  = np.mean(x)
    M1hat  = 0.0
    n      = x.size # sample size
    for ii in range(n): 
        real_ii = ii + 1
        M1hat   = M1hat + x[ii]*(n - real_ii) 
    M1hat = M1hat/(n*(n-1))
    c     = M0hat/gamma( np.log(M0hat/M1hat)/np.log(2)) # scale par
    w     = np.log(2)/np.log(M0hat/(2*M1hat)) # shape par
    return  n, c, w


def wei_fit_pwm_cens(sample, threshold = 0): 
    ''' fit a 2-parameters Weibull distribution to a sample 
    by means of censored Probability Weighted Moments (CPWM) - Wang, 1999
    only observations larger than a value 'threshold' are used for the fit
    but the probability mass of the observations below threshold is accounted for.
    compute the first two PWMs
    ar and br are linear comb of each other, perfectly equivalent
    I use censoring on the br as proposed by Wang 1990
    so that I am censoring the lower part of the distribution
    Default threshold = 0
    INPUT:: sample (array with observations)
           threshold (default is = 0)
    OUTPUT::
    returns numerosity of the sample (n) (only values above threshold)
    Weibull scale (c) and shape (w) parameters '''    
    sample = np.asarray(sample) # from list to Numpy array
    wets   = sample[sample > 0]
    x      = np.sort(wets) # sort ascend by default
    b0  = 0.0
    b1  = 0.0
    n      = x.size # sample size
    for ii in range(n): 
        real_ii = ii + 1
        if x[ii]>threshold:
            b1=b1+x[ii]*(real_ii-1)
            b0=b0+x[ii]
    b1=b1/(n*(n-1))
    b0=b0/n
    # obtain ar=Mrhat  as linear combination of the first two br
    M0hat = b0
    M1hat = b0 - b1
    c     = M0hat/gamma( np.log(M0hat/M1hat)/np.log(2)) # scale par
    w     = np.log(2)/np.log(M0hat/(2*M1hat)) # shape par
    return  n, c, w


def wei_quant(Fi, C, w, ci = False, varcov = []):
    ''' WEI quantiles and (optional) confidence intervals'''
    Fi         = np.asarray(Fi)
    is_scalar  = False if Fi.ndim > 0 else True
    Fi.shape   = (1,)*(1-Fi.ndim) + Fi.shape
    q          = ( -np.log(1-Fi))**(1/w)*C
    q          =  q if not is_scalar else  q[0]
    if ci == True:
        # compute std of quantiles using the DELTA METHOD
        m = np.size(Fi)
        qu = np.zeros(m)
        ql = np.zeros(m)        
        for ii in range(m):
            yr = 1-Fi[ii]
            # dx/dC and dx/dw
            DEL = np.array([ (-np.log(yr))**(1/w),
                   C*(-np.log(yr))**(1/w)*np.log(-np.log(1-Fi[ii])) ])
            prod1 = np.dot(varcov, DEL)
            varz = np.dot( prod1, DEL)    
            stdz = np.sqrt(varz)
            ql[ii] = q[ii] - 1.96*stdz
            qu[ii] = q[ii] + 1.96*stdz            
        qu = qu if not is_scalar else  qu[0]
        ql = ql if not is_scalar else  ql[0]
        return q, qu, ql
    else:
        return q
    
    
def wei_pdf(x,C,W): 
    ''' compute Weibull pdf with parameters scale C and shape w
    for a scalar OR array input of positive values x'''
    x = np.asarray(x) # transform to numpy array
    is_scalar = False if x.ndim > 0 else True # create flag for output
    x.shape = (1,)*(1-x.ndim) + x.shape # give it dimension 1 if scalar
    pdf = W/C*(x/C)**(W - 1)*np.exp(-(x/C)**W )   
    pdf = pdf if not is_scalar else pdf[0]
    return  pdf
   
    
def wei_mean_variance(C,w):
    ''' Computes mean mu and variance var
    of a Weibull distribution with parameter scale C and shape w
    -or repeat for all the elements for same-dim arrays C and W
    NOTE: C and w need to have the same dimension e data type'''
    C = np.asarray(C)
    w = np.asarray(w)  
    # if C is scalar, we return both scalars
    is_C_scalar = False if C.ndim > 0 else True    
    C.shape = (1,)*(1-C.ndim) + C.shape
    w.shape = (1,)*(1-w.ndim) + w.shape
    # compute mean and variance  
    mu    = C/w*gamma(1/w)
    var   = C**2/w**2*(2*w*gamma(2/w)-(gamma(1/w))**2)
    mu    = mu  if not is_C_scalar else mu[0]
    var   = var if not is_C_scalar else var[0]
    return mu,var 


def wei_cdf(q, C, w):# modified order
    ''' returns the non exceedance probability of quantiles q (scalar or array)
    for a Weibull distribution with shape w and scale C'''
    q         = np.asarray(q)
    is_scalar = False if q.ndim > 0 else True 
    cdf       = 1 - np.exp(-(q/C)**w)
    cdf       = cdf  if not is_scalar else cdf[0]
    return cdf


def wei_surv(q, C, w): # modified order
    ''' returns the survival probability of quantiles q (scalar or array)
    for a Weibull distribution with shape w and scale C'''
    q         = np.asarray(q)
    is_scalar = False if q.ndim > 0 else True 
    sdf       = np.exp(-(q/C)**w)
    sdf       = sdf  if not is_scalar else sdf[0]
    return sdf


def wei_random_quant(length,C,w):
    ''' generates a vector of length 'length' of 
    quantiles randomly extracted from a Weibull distr with par C, w
    if length = 1, returns a scalar'''
    Fi = np.random.rand(length)
    xi = ( -np.log(1-Fi))**(1/w)*C
    xi    = xi if length > 1 else xi[0]
    return xi


def wei_fit_ls(sample):
    '''
    fit Weibull distribution to given sample
    removing data that are not positive
    return N (number obs >0)
    C (scale par) and w (shape par)
    '''
    sample  =  np.array(sample)
    sample2 = sample[sample > 0 ]   
    xi      = np.sort(sample2)
    N       = len(xi)
    II      = np.arange(1,N+1)
    Fi      = II/(N+1)
    yr      = np.log( -np.log(1-Fi))
    xr      = np.log(xi)
    
    xrbar   = np.mean(xr)
    yrbar   = np.mean(yr)
    xrstd   = np.std(xr)
    yrstd   = np.std(yr)
    
    w       = yrstd/xrstd # shape par
    C       = np.exp( xrbar-1/w*yrbar) # scale par
    return N, C, w


def wei_fit_mlpy(sample):
    # remove in a future version. the other ml fit works fine
    '''
    fit Weibull using the builtin scipy exponweib function
    setting floc = 0 and fa = 1
    return n size of sample >0 (used for fit)
    '''    
    sample  =  np.array(sample)
    sample2 = sample[sample > 0 ]   
    n       = len(sample2)
    # setting a = 1 (additional generalized exponent)
    # and floc = 0 (zero location here)
    a, w, mu, c = exponweib.fit(sample2, floc=0, fa=1)
    # do not return 2nd shape par. a and loc mu
    return n,c,w   


def wei_fit_ml(sample, std = False):
    '''--------------------------------------------------------------
    fit Weibull by means of Maximum_Likelihood _Estimator (MLE)
    finding numerically the max of the likelihood function
    return n size of sample >0 (used for fit)
    if std = True compute standard deviations and covariances 
    of parameters C and w.
    ----------------------------------------------------------------'''
    sample       =  np.array(sample)
    sample2      = sample[sample > 0 ]   
    x            = sample2
    n            = len(x)    
    # derivative of the log likelihood function with respect with par. w:
    like         = lambda w: n*(1/w-np.sum((x**w)*np.log(x))/ \
                                 np.sum(x**w))+ np.sum(np.log(x))
    w_init_guess = 1.0
    w_hat        = sc.optimize.fsolve(like, w_init_guess )[0]
    c_hat        = ( np.sum(x**w_hat)/n )**(1.0/w_hat)       
    parhat = (c_hat, w_hat)    
    if std:   
        varcov = gev.hess(wei_negloglike, parhat, sample)
        parstd = np.sqrt( np.diag(varcov) )
        return n,c_hat,w_hat , parstd, varcov
    else: 
        return n,c_hat,w_hat 


def wei_negloglike(parhat, data):
    ''' compute Weibull neg log likelihood function
    for a given sample xi and estimated parameters C,w'''
    C = parhat[0]
    w = parhat[1]
    xi   = data[data> 0]
    N    = len(xi)
    nllw = - N*np.log(w/C) -(w-1)*np.sum( np.log(xi/C) ) + np.sum( (xi/C)**w )
    return nllw


###############################################################################
###############################################################################

############################# MEV BASIC FUNCTIONS #############################

###############################################################################
###############################################################################


def mev_fun(y, pr, N, C, W):
    ''' MEV distribution function, to minimize numerically 
    for computing quantiles'''
    nyears = N.size
    mev0f = np.sum( ( 1-np.exp(-(y/C)**W ))**N  ) - nyears*pr 
    return mev0f


def mev_quant(Fi, x0, N, C, W, potmode = True, thresh = 0):
    '''--------------------------------------------------------------------
    computes the MEV quantile for given non exceedance prob. in Fi
    arguments:
    Fi: non exceedance probability (either scalar or array of values)
    x0: starting guess for numerical solution
    N, C, W: Yearly parameters of MEV distribution
    potmode: if True, considers the distributions of value above threshold (default is True)
    (In practice if potmode=True, the distribution of excesses over threshold is computed
    and then from it the cdf is computed for the effective quantile = quant - thresh)
    thresh: threshold for defining ordinary events (default is zero)
    returns:
    quant -> single quantile, or array of quantiles
    flags -> flag = 0 if everything is ok, = 1 if convergence problems
    when It happens, a different x0 should be used.
    ---------------------------------------------------------------------'''
    Fi = np.asarray(Fi)
    is_scalar = False if Fi.ndim > 0 else True  
    Fi.shape = (1,)*(1-Fi.ndim) + Fi.shape    
    m = np.size(Fi)
    quant = np.zeros(m)
    flags = np.zeros((m), dtype = bool) # flag for the convergence of numerical solver
    for ii in range(m):
        myfun     = lambda y: mev_fun(y,Fi[ii],N,C,W)
        res       = sc.optimize.fsolve(myfun, x0, full_output = 1)
        quant[ii] = res[0]
        info      = res[1]
        fval      = info['fvec']
        if fval > 1e-5:
            print('mevd_quant:: ERROR - fsolve does not work -  change x0')
            flags[ii] = 1
        quant  = quant if not is_scalar else quant[0]
        flags  = flags if not is_scalar else flags[0]
    if potmode:
        quant = quant + thresh
    return quant, flags


def mev_cdf(quant, N, C, W, potmode = True, thresh = 0):
    '''----------------------------------------------------------------
    computes the mev cdf (cumulative distribution function):
    given::
        quant: quantile for which I compute the non exceedance probability
        N,C,W: arrays of yearly parameters of the mev distribution
        potmode: if True, considers the distributions of value above threshold (default is True)
            (In practice if potmode=True, the distribution of excesses over threshold is computed
            and then from it the cdf is computed for the effective quantile = quant - thresh)
        thresh: threshold for defining ordinary events (default is zero)
    returns::
        mev_cdf: non exceedance probability for the given quantile
    ----------------------------------------------------------------'''
    quant       = np.asarray(quant)
    is_scalar   = False if quant.ndim > 0 else True
    quant.shape = (1,)*(1-quant.ndim) + quant.shape      
    nyears      = N.shape[0]
    if potmode:
        quant = quant - thresh # Probability for given excess
    m = np.size(quant)
    mev_cdf = np.zeros(m)
    for ii in range (m):
        mev_cdf[ii]     = np.sum( ( 1 - np.exp(-(quant[ii]/C)**W ))**N ) / nyears
    mev_cdf     =  mev_cdf  if not is_scalar else  mev_cdf[0]
    return mev_cdf


def mev_fit(df, ws = 1, how = 'pwm', threshold = 0, potmode = True, declu = False):
    '''--------------------------------------------------------------------
    fit MEV to a dataframe of daily rainfall observations df - with PRCP, YEAR fields
    fitting Weibull to windows of size ws (scalar integer value, default is 1)
    ws = window size in years (default is 1)
    how = fitting method. available are 'ml', 'pwm', 'ls' (default is 'pwm')
    (for 'ml'=maximum likelihood, 'pwm'=probability weighted moments, 'ls'=least squares)
    potmode: if True, considers the distributions of value above threshold (default is True)
    (In practice if potmode=True, the distribution of excesses over threshold is computed
    and therefore N,C,W parameters refer to the distribution of excesses over threshold.
    threshold: threshold for defining ordinary events (default is zero)
    declu: If True, before fitting WEI decluster time series so that only indep events are used for fitting MEV
    (computed globally for the whole time series)
    compute the lag at which the correlation decays at the 90% percentile of the noise level
    and keep only the largest value within that distance
    return::
    N,C,W = arrays of Weibull parameters for each year/block
    (shape -> arrays nwinsizes * nyears)
    --------------------------------------------------------------------------'''
    if declu:
        # decluster time series, and subs. it in the original data frame.
        # (dome together for the entire time series)
        sam = np.array(df.PRCP)
        new_sam = decluster(sam)[0]
        df.PRCP = new_sam

    years   = np.unique(df.YEAR)
    nyears  = np.size(years)
    datamat = np.zeros((nyears, 366))
    for ii in range(nyears):
        datayear = np.array( df.PRCP[df['YEAR'].astype(int) == years[ii]])
        for jj in range(len(datayear)):
            datamat[ii, jj] = datayear[jj]
    # check window is not longer than available sample        
    if ws > nyears:
        print('''mev_fit WARNING: the selected window size is larger than 
              the available sample. Using instead only one window with all
              years available. please check''')
        ws = nyears
    winsize = np.int32( ws ) 
    numwind = nyears // winsize
    ncal2   = numwind*winsize
    datamat_cal_2 = datamat[:ncal2, :]
    wind_cal = datamat_cal_2.reshape( numwind, 366*winsize)
    Ci = np.zeros(numwind)
    Wi = np.zeros(numwind)
    for iiw in range(numwind): # loop on windows of a given size
        sample = wind_cal[iiw, :] 
        # print('how = ', how)
        if not potmode:
            temp, Ci[iiw], Wi[iiw] = wei_fit(sample, how = how, threshold = threshold)
        else:
            excesses = sample[sample > threshold] - threshold
            temp, Ci[iiw], Wi[iiw] = wei_fit(excesses, how = how, threshold = 0)
    N = np.zeros(ncal2)
    for iiw in range(ncal2):
        sample = datamat_cal_2[iiw,:]
        wets   = sample[sample > threshold]
        N[iiw] = np.size(wets)
    C = np.repeat(Ci, winsize)
    W = np.repeat(Wi, winsize)
    return N,C,W


def mev_CI(df, Fi_val, x0, ws = 1, ntimes = 1000, MEV_how = 'pwm', 
        MEV_thresh = 0.0, std_how = 'boot', potmode = True, declu = False):
    '''-----------------------------------------------------------------------
    non parametric bootstrap technique for MEV
    given an observed sample, at every bootstrap iteration generates a new sample with replacement
    and then for each generated sample fits mev and obtains a quantile estimate
    :arguments:
    df = dataframe with precipitation data, where missing data are assumed to be already taken care of
    Fi_val = value of non exceedance probability for which quantile is estimated
    ntimes: number of bootstrap resampling times (default 1000). It should b elarge, at least 20 to allow to compute
    empirical MEV 95% CI in hyp. estimates are normal distribution
    MEV_how: fitting method for WEI-MEV (default 'pwm' probability weighted moments)
    MEV_thresh: threshold for selecting ordinary data. (default threshold = 0)
    potmode: if True, fit WEI to excesses above threshold
             if False, fit WEI to entire values, neglecting observations below threshold (default is True)
    declu: If True, before fitting WEI decluster time series so that only indep events are used for fitting MEV
    (computed globally for the whole time series)
    POSSIBLE OPTIONS FOR COMPUTING CONFIDENCE INTERVALS:
    ---------------------------------------------------------------------------
    std_how = 'boot': do non-parametric bootstrapping for daily values
                      and number of events/year
                      NB: This might reduce variability
                      then hyp. normal distribution of quantiles
                      (default)
    std_how = 'delta': use delta method under the hyp. that all are independent parameters
                       and compute their individual effects on MEV quantiles
    std_how = 'boot_cdf': as before, but without normality assumption. But I need
                         ntimes large enough to compute prob of 95% and 5%.
    std_how = 'par': only resample from the arrays of (Ni, Ci, Wi) for each year/window -
                (default is 'boot')
                Only 'boot' and 'boot_par' available
    ---------------------------------------------------------------------------
    Returns:
        Q_est: 2D array of estimated quantiles (mean over ntimes realizations)
        Q_up: 2D array of quantiles CI upper limit (mean + 2* stdv over ntimes realizations)
        Q_low: 2D array of quantiles CI lower limit (mean - 2* stdv over ntimes realizations)
        Flags: 2D array (ntimes*num_quantiles) with flag for converge of numeric solution
        if = 0 everything converge, if = 1 convergence not achieved.
    -------------------------------------------------------------------------'''
    if declu:
        # decluster time series, and subs. it in the original data frame.
        # (dome together for the entire time series)
        sam = np.array(df.PRCP)
        new_sam = decluster(sam)[0]
        df.PRCP = new_sam

    Fi_val       = np.asarray(Fi_val)
    is_scalar   = False if Fi_val.ndim > 0 else True
    Fi_val.shape = (1,)*(1-Fi_val.ndim) + Fi_val.shape
    m = np.size(Fi_val) 
    QM = np.zeros((ntimes, m))
    Flags = np.zeros((ntimes, m))

    N, C, W = mev_fit(df, ws = ws, how = MEV_how, threshold = MEV_thresh, potmode = potmode)
    Q_est, flags = mev_quant(Fi_val, x0, N, C, W, potmode = potmode, thresh = MEV_thresh)

#    if std_how == 'hess': # NOT SURE IT IS OK - IMPLIES INDEP PARAMETERS
#        print('mev_CI ERROR: method "hess" not available yet')
        
#    if std_how == 'boot_all': # UNDERESTIMATE MEAN BC OF REDUCED VARIABILITY
#        for ii in range(ntimes):
#            print('mev_CI - boot ntimes:', ii ,'/', ntimes)
#            dfr = mev_boot(df) # resample daily data
#            N, C, W = mev_fit(dfr, ws, how = MEV_how, threshold = MEV_thresh)
#            QM[ii,:] = mev_quant(Fi_val, x0, N, C, W)
#        Q_up = np.zeros(m)
#        Q_low = np.zeros(m)
#        for jj in range(m):
#            qi = np.sort( QM[:,jj]) # sort ascend
#            # if CI around true value
#            Q_up[jj] = Q_est[jj] + 1.96*np.std(qi)
#            Q_low[jj] = Q_est[jj] - 1.96*np.std(qi)
##            Q_up[jj]  = np.mean(qi) + 1.96*np.std(qi)
##            Q_low[jj] = np.mean(qi) - 1.96*np.std(qi)
##            Q_est[jj] = np.mean(qi)
            
    if std_how == 'boot': # UNDERESTIMATE MEAN BC OF REDUCED VARIABILITY
        for ii in range(ntimes):
            # print('mev_CI - boot ntimes:', ii ,'/', ntimes)
            dfr = mev_boot_yearly(df) # resample daily data
            N, C, W = mev_fit(dfr, ws, how = MEV_how, threshold = MEV_thresh, potmode = potmode)
            QM[ii,:], Flags[ii, :] = mev_quant(Fi_val, x0, N, C, W, potmode = potmode, thresh = MEV_thresh)
        Q_up = np.zeros(m)
        Q_low = np.zeros(m)
        for jj in range(m):
            qi = np.sort( QM[:,jj]) # sort ascend
            # if CI around true value
            Q_up[jj] = Q_est[jj] + 1.96*np.std(qi)
            Q_low[jj] = Q_est[jj] - 1.96*np.std(qi)
#             Q_up[jj]  = np.mean(qi) + 1.96*np.std(qi)
#             Q_low[jj] = np.mean(qi) - 1.96*np.std(qi)
#             Q_est[jj] = np.mean(qi)

    if std_how == 'boot_cdf': # TO CHECK
        fi = np.arange(1, ntimes + 1)/(ntimes + 1)
        for ii in range(ntimes):
            dfr = mev_boot_yearly(df) # resample daily data
            N, C, W = mev_fit(dfr, ws, how = MEV_how, threshold = MEV_thresh, potmode=potmode)
            QM[ii,:], Flags[ii, :] = mev_quant(Fi_val, x0, N, C, W, potmode = potmode, thresh = MEV_thresh)
        Q_up = np.zeros(m)
        Q_low = np.zeros(m)
        for jj in range(m):
            qi = np.sort( QM[:,jj]) # sort ascend
            Q_up[jj] = np.min( qi[fi > 0.95])
            Q_low[jj] = np.max( qi[fi < 0.05])
            # Q_est[jj] = np.mean(qi)
            
#    if std_how == 'par': # only resample yearly parameters N,C,W
#        npar =np.arange(np.size(N))
#        for ii in range(ntimes):
#            # resample from WEIBULL parameters
#            index = np.random.choice(npar)
#            Nr = N[index]
#            Cr = C[index]
#            Wr = W[index]
#            QM[ii,:] = mev_quant(Fi_val, x0, Nr, Cr, Wr)
#        Q_up = np.zeros(m)
#        Q_low = np.zeros(m)
#        for jj in range(m):
#            qi = np.sort( QM[:,jj]) # sort ascend
#            # if CI around true value
#            Q_up[jj] =  Q_est[jj] + 1.96*np.std(qi)
#            Q_low[jj] = Q_est[jj] - 1.96*np.std(qi)
##            Q_up[jj]  = np.mean(qi) + 1.96*np.std(qi)
##            Q_low[jj] = np.mean(qi) - 1.96*np.std(qi)
##            Q_est[jj] = np.mean(qi)
    
    # now it returns arrays or scalar depending of the type of arguments:
    Q_up    =  Q_up    if not is_scalar else  Q_up[0]
    Q_low   =  Q_low   if not is_scalar else  Q_low[0]
    Q_est  =  Q_est  if not is_scalar else  Q_est[0]
    Flags = Flags if not is_scalar else Flags[0]
    return Q_est, Q_up, Q_low, Flags
        
        
#def mev_boot(df):
#    ''' non parametric bootstrap technique for MEV
#    reshuffle i) the number of events for each year in the series
#    ii) the daily events. For each year generates N_i events.
#    For both steps, we sample with replacement.'''
#    ndays = 366
#    years   = np.unique(df.YEAR)
#    nyears  = np.size(years)
#    datamat = np.zeros((nyears, ndays))
#    datayear = np.zeros((nyears,ndays))
#    Ni = np.zeros(nyears)
#    for ii in range(nyears):
#        samii = df.PRCP[df.YEAR == years[ii]]
#        Ni[ii] = np.size( samii[samii > 0.0])                
#    sample  = df.PRCP
#    for ii in range(nyears):
#        ni = np.int32( np.random.choice(Ni)) # extract a number of wet events
#        # print(ni)
#        # print(Ni)
#        # print('test')
#        # print(np.random.choice(sample, size = ni))
#        datamat[ii,:ni] = np.random.choice(sample, size = ni) # extract the events
#        datayear[ii,:] = np.repeat(years[ii], ndays) 
#    prcp = datamat.flatten()
#    year = datayear.flatten()
#    mydict = { 'YEAR' : year, 'PRCP' : prcp}
#    dfr  = pd.DataFrame(mydict)
##    print('dfr')
##    print(dfr.head())
##    print('df')
##    print(df.head())
##    sys.exit()
#    return dfr


def mev_boot_yearly(df):
    ''' non parametric bootstrap technique for MEV
    reshuffle i) the number of events for each year in the series
    ii) the daily events. For each year generates N_i events.
    For both steps, we sample with replacement.'''
    ndays = 366
    years   = np.unique(df.YEAR)
    nyears  = np.size(years)
    datamat = np.zeros((nyears, ndays))
    datamat_r =  np.zeros((nyears, ndays))
    datayear = np.zeros((nyears,ndays))
    Ni = np.zeros(nyears, dtype = np.int32)
    indexes = np.arange(nyears)
    
    for ii in range(nyears):
        samii = df.PRCP[df.YEAR == years[ii]]
        wetsii = samii[samii > 0.0]
        Ni[ii] = np.size( wetsii )  
        # print('Niiii = ', Ni[ii])
        datamat[ii, :Ni[ii]] = wetsii  
    
    # resample daily values for every year
    for ii in range(nyears):    
        myind = np.random.choice(indexes) # sample one year at random, with its N and daily values
        original = datamat[myind, :]
        datamat_r[ii,:] = np.random.choice( original , size = ndays)
        datayear[ii,:] = np.repeat(years[ii], ndays) 
        
    prcp = datamat_r.flatten()
    year = datayear.flatten()
    mydict = { 'YEAR' : year, 'PRCP' : prcp}
    dfr  = pd.DataFrame(mydict)
#    print('dfr')
#    print(dfr.head())
#    print('df')
#    print(df.head())
#    sys.exit()
    return dfr






###############################################################################
###############################################################################
    
##################  DATA ANALYSIS MAIN FUNCTIONS ##############################
    
###############################################################################
###############################################################################  


def decluster(sample, noise_lag = 10, noise_prob = 0.9, max_lag = 30):
    '''------------------------------------------------------------
    Decluster a rainfall time series. Eliminate the non-zero value within a correlation window
    Once correlation time scale is compute, running windows with its size are selected.
    within each of them, I only keep the largest event observed.
    :param sample: the time series to be declustered.
    :param noise_lag: minimum lag at which I consider the ACF to be noise (defult = 10)
    :param noise_prob: assume as noise correlation the 75% percentile of values beyond noise_lag
    :param max_lag: max lag used to compute the noise level (default = 30)
    :return:
    -> dec_sample -> declustered time series
    -> dec_lag    -> running lag window size in which I only keep the maximum
    --------------------------------------------------------------'''
    det_sample = sample.copy()
    # noise_lag = 6 # number of days after which I assume the time series to be perfectly scorrelated
    # noise_prob = 0.75 # quantile of correlation that we assume to represent noise
    lags = np.arange(max_lag) # 20 - days
    nlags = np.size(lags)
    time_corr = np.zeros(nlags)
    n = np.size(sample)
    for iil in lags:
        mylag = lags[iil]
        xt = sample[:n-mylag]
        xtpt = sample[mylag:n]
        time_corr[iil] = pearsonr(xt, xtpt)[0]
    noise = time_corr[noise_lag:]
    sorted_noise = np.sort(noise) # ascend by default
    num_noise = np.size(noise)
    fi = np.arange(num_noise+1)/(num_noise+1) # non exceedance
    mypos = np.argmin( np.abs(fi-noise_prob))
    noise_level = sorted_noise[mypos]
    # first crossing of noise level
    racf = time_corr - noise_level
    # now find its first zero crossing
    zero_crossings = np.where(np.diff(np.sign(racf)))[0]
    dec_par = zero_crossings[0]
    # mylags = np.arange(1, dec_par + 1)
    for iis in range(n-dec_par):
        window = sample[iis:iis+dec_par+1]
        loc_max = np.max(window)
        for iie in range(np.size(window)):
            if window[iie] < loc_max:
                det_sample[iis + iie] = 0
    return det_sample, dec_par


def remove_missing_years(df, nmin):
    '''
    # input has to be a pandas data frame df
    # including the variables YEAR, PRCP
    # returns the same dataset after removing all years with less of nmin days of data
    # (accounts for missing entries, negative values)
    # the number of years remaining (nyears2)
    # and the original number of years (nyears1)
    '''
    years_all  = df['YEAR']
    years      = pd.Series.unique(years_all)
    nyears1    = np.size(years)
    for jj in range(nyears1):
        dfjj      = df[ df['YEAR'] == years[jj] ]
        my_year   = dfjj.PRCP[ dfjj['PRCP'] >= 0 ] # remove -9999 V
        my_year2  = my_year[ np.isfinite(my_year) ] # remove  nans - infs V
        my_length = len(my_year2)
        if my_length < 366-nmin:
            df    = df[df.YEAR != years[jj]] # remove this year from the data frame
    # then remove NaNs and -9999 from the record
    # df.dropna(subset=['PRCP'], inplace = True)
    df = df.dropna(subset=['PRCP'])
    df = df.ix[df['PRCP'] >= 0]
    # check how many years remain      
    years_all_2 = df['YEAR']    
    nyears2 = np.size(pd.Series.unique(years_all_2))
    return df, nyears2, nyears1


def tab_rain_max(df):
    '''--------------------------------------------------------------------------
    arguments: df, pandas data frame with fields YEAR, PRCP
    returns:
    XI -> array of annual maxima (ranked in ascending order)
    Fi -> Weibull plotting position estimate of their non exceedance probability
    TR -> their relative return times
    Default using Weibull plotting position for non exceedance probability
    -----------------------------------------------------------------------------'''
    years_all  = df['YEAR']
    years      = np.unique(years_all)
    nyears     = np.size(years)
    maxima     = np.zeros(nyears)
    for jj in range(nyears):
        my_year      = df.PRCP[df['YEAR'] == years[jj]]
        maxima[jj]   = np.max(my_year)
    XI         = np.sort(maxima, axis = 0) # default ascend
    Fi         = np.arange(1,nyears+1)/(nyears + 1)
    TR         = 1/(1 - Fi)  
    return XI,Fi,TR


def table_rainfall_maxima(df, how = 'pwm', thresh = 0, potmode = True, declu = False):
    '''--------------------------------------------------------------------------
    arguments:
    df, pandas data frame with fields YEAR, PRCP
    how = method for fitting Weibull (default 'pwm' for probability weighted moments)
    thresh: threshold for selecting ordinary data. (default threshold = 0)
    potmode: if True, fit WEI to excesses above threshold
             if False, fit WEI to entire values, neglecting observations below threshold (default is True)
     declu: If True, before fitting WEI decluster time series so that only indep events are used for fitting MEV
    (computed globally for the whole time series)
    returns:
    XI -> array of annual maxima (ranked in ascending order)
    Fi -> Weibull plotting position estimate of their non exceedance probability
    TR -> their relative return times
    NCW -> Array of shape nyears*3 with yearly parameters values in order N, C, W.
    ## you need to take care of missing values / nans before using this function, use
    ## remove missing years.
    -----------------------------------------------------------------------------'''

    if declu:
        # decluster time series, and subs. it in the original data frame.
        # (dome together for the entire time series)
        sam = np.array(df.PRCP)
        new_sam = decluster(sam)[0]
        df.PRCP = new_sam

    years_all  = df['YEAR']
    years      = pd.Series.unique(years_all)
    nyears     = len(years)
    maxima     = np.zeros([nyears, 1])
    NCW        = np.zeros([nyears, 3])
    for jj in range(nyears):
        my_year       = df.PRCP[df['YEAR'] == years[jj]]
        maxima[jj, 0] = np.max(my_year)
        if potmode:
            excesses = my_year[my_year > thresh] - thresh
            (NCW[jj, 0], NCW[jj, 1], NCW[jj, 2]) = wei_fit(excesses , how = how,
                                                           threshold = 0)
        else:
            (NCW[jj, 0], NCW[jj, 1], NCW[jj, 2]) = wei_fit(my_year , how = how,
                                               threshold = thresh)
    XI = np.sort(maxima, axis = 0) # default ascend
    Fi = np.arange(1, nyears + 1)/(nyears + 1)
    TR = 1/(1 - Fi)  
    return XI, Fi, TR, NCW


def fit_EV_models(df, tr_min = 5, ws = 1, GEV_how = 'lmom', MEV_how = 'pwm', 
                  MEV_thresh = 0, MEV_potmode = True, POT_way = 'ea', POT_val = 3, POT_how = 'ml',
        ci = False, ntimes = 1000, std_how_MEV = 'boot', std_how_GEV = 'hess', 
                                            std_how_POT = 'hess', rmy = 36, declu = False):
    ''' -------------------------------------------------------------------------------
    fit MEV, GEV and POT to daily data in the dataframe df
    with fields PRCP and YEAR, and compare them with observed annual maxima
    compute quantiles - and non exceedance probabilities
    and compare with the same dataset / produce QQ and PP plots
    default methods are PWM, LMOM, and ML for MEV-GEV-POT respectively
    MEV - fit Weibull to windows of size ws, default ws = 1 (yearly Weibull)

    declu: If True, before fitting WEI decluster time series so that only indep events are used for fitting MEV
    (computed globally for the whole time series)
    -----------------------------------------------------------------------------------'''

    df, ny2, ny1 = remove_missing_years(df, rmy)

    if declu:
        # decluster time series, and subs. it in the original data frame.
        # (dome together for the entire time series)
        sam = np.array(df.PRCP)
        new_sam = decluster(sam)[0]
        df.PRCP = new_sam

    XI,Fi,TR     = tab_rain_max(df)
    tr_mask      = TR > tr_min
    TR_val       = TR[tr_mask]
    XI_val       = XI[tr_mask]
    Fi_val       = Fi[tr_mask]

    #x0           = np.mean(XI_val) - 0.2*np.std(XI_val)
    x0 = 50.0

    if declu:
        # decluster time series, and subs. it in the original data frame.
        # (dome together for the entire time series)
        sam = np.array(df.PRCP)
        new_sam = decluster(sam)[0]
        df.PRCP = new_sam

    # fit distributions
    csi, psi, mu    = gev.gev_fit(XI, how = GEV_how)
    N, C, W         = mev_fit(df, ws= ws, how = MEV_how, threshold = MEV_thresh, potmode = MEV_potmode)
    csip, psip, mup = gev.pot_fit(df, datatype = 'df', way = POT_way, ea = POT_val,
                               sp = POT_val, thresh = POT_val,  how = POT_how)
    # compute quantiles
    QM, flags    = mev_quant(Fi_val, x0, N, C, W, potmode = MEV_potmode, thresh = MEV_thresh)
    QG           = mu + psi/csi*(( -np.log(Fi_val))**(-csi) -1)
    QP           = mup + psip/csip*(( -np.log(Fi_val))**(-csip) -1)

    # compute non-exceedance frequencies
    FhM          = mev_cdf(XI_val,N,C,W, potmode = MEV_potmode, thresh = MEV_thresh)
    FhG          = gev.gev_cdf(XI_val, csi, psi, mu)
    FhP          = gev.gev_cdf(XI_val, csip, psip, mup)
    
    if ci:
        # MEV - re-evaluate the mean here!
        QmM, QuM, QlM, Flags = mev_CI(df, Fi_val, x0, ws = ws, ntimes = ntimes,
                               MEV_how = MEV_how, MEV_thresh = MEV_thresh , std_how = std_how_MEV, potmode = MEV_potmode)        
        # POT
        parhat_POT, parpot_POT, parstd_POT, varcov_POT = gev.pot_fit(df, datatype = 'df', way = POT_way, ea = POT_val, 
                  sp = POT_val, thresh = POT_val, how = POT_how, std = True, std_how = std_how_POT, std_num = ntimes)
        QmP, QuP, QlP = gev.pot_quant(Fi_val, csip, psip, mup, ci = True, parpot = parpot_POT, varcov = varcov_POT)
        # GEV
        parhat_GEV, parstd_GEV, varcov_GEV = gev.gev_fit(XI, how = GEV_how,
                        std = True, std_how = std_how_GEV, std_num = ntimes)
        QmG, QuG, QlG = gev.gev_quant(Fi_val, csi, psi, mu, ci = True, varcov = varcov_GEV)
        
        return TR_val, XI_val, Fi_val, QM, QG, QP, QuM, QuG, QuP, QlM, QlG, QlP, FhM, FhG, FhP, flags
        # return TR_val, XI_val, Fi_val, QM, QG, QP, QmM, QmG, QmP, QuM, QuG, QuP, QlM, QlG, QlP, FhM, FhG, FhP
    else:
        return TR_val, XI_val, Fi_val, QM, QG, QP, FhM, FhG, FhP, flags
    

def shuffle_mat(datamat, ncal, nval):
    # this only shuffles YEARS in a sample of daily rainfall -
    # for cross validation of extreme value models
    '''given an array with shape (nyears*ndays)
    scramble its years
    and returns a calibration martrix, calibration maxima,
    and independent validation maxima'''

    # number of wet days for each year
    nyears = datamat.shape[0]

    randy     = np.random.permutation( int(nyears) ) 
    datamat_r = datamat[randy]
    mat_cal   = datamat_r[:ncal, :]
    maxima    = np.max(datamat_r, axis = 1) # yearly maxima
    max_cal   = maxima[:ncal]
    max_val   = maxima[ncal:ncal+nval]
    return mat_cal, max_cal, max_val, datamat_r


def shuffle_all(datamat, ncal, nval):
    # this only shuffles N and all daily values -
    # as in Zorzetto et al, 2016
    # for cross validation of extreme value models
    '''given an array with shape (nyears*ndays)
    scramble its years
    and returns a calibration martrix, calibration maxima,
    and independent validation maxima'''
    
    # number of wet days for each year
    nyears = datamat.shape[0]
    ndays = datamat.shape[1] # should be 366
    # print('nyears = ', nyears)
    # print('ndays = ', ndays)
    # shuffle all daily data
    all_data = datamat.flatten()
    all_wets = all_data[all_data > 0]
    all_rand = np.random.permutation(all_wets)
    
    # get number of wet days / year
    nwets = np.zeros(nyears)
    for ii in range(nyears):
        sample = datamat[ii,:]
        nwets[ii] = np.size( sample[sample > 0]) # yearly number of wet days
    
    # shuffle yearly number of wet days
    nwets_rand = np.random.permutation(nwets)
    
    # fill a new array with the reshuffled data
    datamat_r = np.zeros((nyears, ndays))
    count = 0
    for ii in range(nyears):
        ni = np.int32(nwets_rand[ii])
        datamat_r[ii, :ni] = all_rand[count:count+ni]
#        print('start = ', count)
#        print('end = ', count + ni)
        count = count + ni
# get calibration and validation samples    
    mat_cal   = datamat_r[:ncal, :]
    maxima    = np.max(datamat_r, axis = 1) # yearly maxima
    max_cal   = maxima[:ncal]
    max_val   = maxima[ncal:ncal+nval]
    return mat_cal, max_cal, max_val, datamat_r


def cross_validation(df, ngen, ncal, nval, tr_min = 5, ws=[1],
                     GEV_how = 'lmom', MEV_how = 'pwm', MEV_thresh = 0, MEV_potmode = True,
                 POT_way = 'ea', POT_val = [3], cross = True, ncal_auto = 100,
                 shuff = 'year', declu = False):
    '''------------------------------------------------------------------------
    FIT MEV and GEV and perform validation with stationary time series
    obtained reshuffling the years of the original time series
    ## you need to take care of missing values / nans before using this function,
    ## use 'remove missing years'.
    ###########################################################################
    INPUT::
        df -  data frame with fields 'PRCP' daily precipitation values (float)
                                    'YEAR' year in format yyyy (integer/float)
        ngen - number of random re-shuffling of the dataset
        ncal - nyears of data for calibration (used in cross - mode only)
        nval - nyears of data for validation  (used in cross - mode only)
        
        tr_min - minimum return time for which I compute quantiles
                                 default is 5 years
                                 (to avoid values too low or too close to 1)
        ws - array of windows [in years] used to fit block-Weibull for MEV
                             (default is 1 year - yearly Weibull parameters)
        MEV_how - fitting method for Weibull. 
                   options are: 'pwm' - Probability Weighted Moments (default)
                                'ls'  - Least Squares
                                'ml'  - Maximum Likelihood
                                
        MEV_thresh - optional threshold for MEV. only works for how = 'pwm'
                     probability mass below threshold is just ignored.
                     (default value is zero)
                     
        MEV_potmode - If True, consider pdf of daily rainfall with area 1 above threshold 
                    (default = True)
                     
        POT_way - threshold selection method for POT. can be:
            'ea' fixed number average exceedances / year
            'sp' survival probability to be beyond threshold 
            'thresh' value of the threshold
            
        POT_val - value assigned to the threshold. 
             depending on the value of POT_way, 
             it is the value for 'ea', 'sp' or 'thresh'
                         
        cross - if True, use cross validation for evaluating model performance
                at each reshuffling calibration and validation on ind. samples
                
                if False, use the same interval of years 
                for calibration and validation   
        ncal_auto - (only used when cross = False), this is the legth [years]
                    of the same sample used for both calibration & validation
        
        shuff = 'all' -> reshuffle daily values as in Zorzetto et al, 2016  
              = 'year' -> only resample years with resubstitution          
    ###########################################################################          
    OUTPUT::
        TR_val - array of return times for which quantiles are computed
                
        m_rmse - MEV root mean squared error (for est. quantiles)
            (array with shape:  nwinsizes * ntr )   
        g_rmse - GEV root mean squared error (for est. quantiles)
            (array with shape: ntr  ) 
        p_rmse - POT root mean squared error (for est. quantiles)
            (array with shape: ntr  )         
        em - MEV relative errors
            (array with shape:  ngen * nwinsizes * ntr )   
        eg - GEV relative errors
            (array with shape:  ngen * ntr )  
        eg - POT relative errors
            (array with shape:  ngen * ntr )
    ########################################################################'''

    if declu:
        # decluster time series, and subs. it in the original data frame.
        # (dome together for the entire time series)
        sam = np.array(df.PRCP)
        new_sam = decluster(sam)[0]
        df.PRCP = new_sam

    years   = np.unique(df.YEAR)
    nyears  = np.size(years)
    datamat = np.zeros((nyears, 366))
    for ii in range(nyears):
        datayear = np.array( df.PRCP[df['YEAR'].astype(int) == years[ii]])
        for jj in range(len(datayear)):
            datamat[ii, jj] = datayear[jj]
    
    # for cross validation  
    if cross == True:         
        Fi_val0     = np.arange(1,nval+1)/(nval+1) # Weibull plotting position
        TR_val0     = 1/(1-Fi_val0)
        index_tr   = TR_val0 > tr_min
        TR_val     = TR_val0[index_tr]
        Fi_val     = Fi_val0[index_tr]
    else: # for same - sample validation
        Fi_val0     = np.arange(1,ncal_auto+1)/(ncal_auto+1) # Weibull plotting position
        TR_val0     = 1/(1-Fi_val0)
        index_tr   = TR_val0 > tr_min
        TR_val     = TR_val0[index_tr]
        Fi_val     = Fi_val0[index_tr]        
        
    ntr       = np.size(TR_val)
    nwinsizes = np.size(ws)   
    nthresh   = np.size(POT_val)

    Flags     = np.zeros((ngen, nwinsizes, ntr))
    em        = np.zeros((ngen, nwinsizes, ntr))
    eg        = np.zeros((ngen, ntr))
    ep        = np.zeros((ngen, nthresh, ntr))
    m_rmse    = np.zeros((nwinsizes, ntr))
    g_rmse    = np.zeros(ntr)
    p_rmse    = np.zeros((nthresh, ntr))
     
    for iig in range(ngen): # loop on random generations
        
        if shuff == 'year': # resample years only, or
            mat_cal, max_cal, max_val, datamat_r = shuffle_mat(datamat, ncal, nval)
        elif shuff == 'all': # or reshuffle daily values
            mat_cal, max_cal, max_val, datamat_r = shuffle_all(datamat, ncal, nval)
        if cross == True:            
            XI_val0    = np.sort(max_val, axis = 0)
            XI_val     = XI_val0[index_tr] 
        else: # same - sample validation and calibration
            nval    = ncal_auto
            ncal    = ncal_auto
            max_cal = np.max(datamat_r, axis = 1)[:ncal_auto]
            max_val = max_cal
            mat_cal = datamat_r[:ncal_auto, :]
            XI_val0 = np.sort(max_val)
            XI_val  = XI_val0[index_tr] 

        # fit GEV and compute errors
        csi, psi, mu = gev.gev_fit(max_cal, how = GEV_how)
        QG  = mu + psi/csi*(( -np.log(Fi_val))**(-csi) -1) # for every Tr
        for iitr in range(ntr): # compute gev relative errors
            eg[iig, iitr]  = (QG[iitr] - XI_val[iitr])/XI_val[iitr]
            
        # fit POT and compute errors - pass inputs in the future
        for iith in range(nthresh):
            potval = POT_val[iith]
            csip, psip, mup = gev.pot_fit(mat_cal, datatype = 'mat', 
                   way = POT_way, ea = potval, sp = potval, thresh = potval, how = 'ml')
            QP  = mup + psip/csip*(( -np.log(Fi_val))**(-csip) -1) # for every Tr
        
            for iitr in range(ntr): # compute pot relative errors
                ep[iig, iith, iitr]  = (QP[iitr] - XI_val[iitr])/XI_val[iitr]
            
        # fit MEV for blocks of differing size
        x0 = np.mean(max_cal) # mev quantile first guess / change it if needed
        # print(ws)
        for iiws in range(nwinsizes):
            winsize = np.int32( ws[iiws]) 
                
            # check window is not longer than available sample      
            if winsize > ncal:
                print('''cross_validation WARNING: 
                      at least on of the selected window sizes is larger than 
                      the calibration sample. Using instead only one window with all
                      years available. please check''')
                winsize = ncal
        
            numwind = ncal // winsize
            ncal2 = numwind*winsize
            datamat_cal_2 = mat_cal[:ncal2, :]
            wind_cal = datamat_cal_2.reshape( numwind, 366*winsize)
            
            Ci = np.zeros(numwind)
            Wi = np.zeros(numwind)
            for iiw in range(numwind): # loop on windows of a given size
                sample = wind_cal[iiw, :]                
                # compute the global Weibull parameters
                if MEV_potmode:
                    excesses = sample[sample > MEV_thresh]- MEV_thresh
                    temp, Ci[iiw], Wi[iiw]  = wei_fit(excesses , how = MEV_how, 
                                                      threshold = 0)    
                else:
                    temp, Ci[iiw], Wi[iiw]  = wei_fit(sample , how = MEV_how, 
                                                      threshold = MEV_thresh)                
            N = np.zeros(ncal2)
            for iiw in range(ncal2):
                sample = datamat_cal_2[iiw,:]
                wets = sample[sample > MEV_thresh] # Ok for potmode as well - N is the same above threshold
                N[iiw]=np.size(wets)
                
            C = np.repeat(Ci, winsize)
            W = np.repeat(Wi, winsize)
            QM, flags = mev_quant(Fi_val, x0, N, C, W, potmode = MEV_potmode, thresh = MEV_thresh)

            Flags[iig, iiws, :] = flags
            em[iig, iiws, :] = (QM - XI_val) /XI_val

            # for iitr in range(ntr): # comput emev relative errors
            #     em[iig, iiws, iitr] = (QM[iitr] - XI_val[iitr])/XI_val[iitr]
            #     Flags[iig, iiws, :] = flags
                
    # compute root mean squared errors (RMSE)       
    for iitr in range(ntr):
        egt = eg[:, iitr].flatten()
        g_rmse[iitr]  =  np.sqrt(  np.mean( egt**2 ))  
        
    for iitr in range(ntr):
        for iith in range(nthresh):
            ept = ep[:, iith, iitr].flatten()
            p_rmse[iith, iitr]  =  np.sqrt(  np.mean( ept**2 )) 
    
    for iitr in range(ntr):
        for iiws in range(nwinsizes):
            emt = em[:, iiws, iitr].flatten()
            m_rmse[iiws, iitr]  =  np.sqrt(  np.mean( emt**2 ))
    return TR_val, m_rmse, g_rmse, p_rmse, em, eg, ep, Flags


def slideover(df, winsize = 30, Tr = 100, display = True, ci = True, ntimes = 100):
    ''' perform EV analysis on sliding and overlapping windows
    declustering and threshold not supported here - for now'''
    years = np.unique(df.YEAR)
    nyears = np.size(years)
    nwin = nyears - winsize + 1    
    mq = np.zeros(nwin)
    mqu = np.zeros(nwin)
    mql = np.zeros(nwin)    
    gq = np.zeros(nwin)
    gqu = np.zeros(nwin)
    gql = np.zeros(nwin)    
    pq = np.zeros(nwin)
    pqu = np.zeros(nwin)
    pql = np.zeros(nwin)
    central_year = np.zeros(nwin)
    for ii in range(nwin):
        print('slideover _ window = ', ii, 'of', nwin)        
        wyears = years[ii:ii + winsize]
        central_year[ii] = np.rint(np.mean(wyears))
        df1 = df[ (df['YEAR'] >= wyears[0]) & (df['YEAR'] < wyears[-1])]
        XI,Fi,TR = tab_rain_max(df1)        
        if ci == True:
            parhat_GEV, parstd_GEV, varcov_GEV = gev.gev_fit(XI, 
                how = 'lmom', std = True, std_how = 'hess', std_num = ntimes)
            csi, psi, mu = parhat_GEV
            gq[ii], gqu[ii], gql[ii] = gev.gev_quant(1-1/Tr, 
                           csi, psi, mu, ci = True, varcov = varcov_GEV)
            
            parhat_POT, parpot_POT, parstd_POT, varcov_POT = gev.pot_fit(df1, 
                                datatype = 'df', way = 'ea', ea = 3, 
                        how = 'ml', std = True, std_how = 'hess', std_num = ntimes)
            csip, psip, mup = parhat_POT
            pq[ii], pqu[ii], pql[ii] = gev.pot_quant(1-1/Tr, csip, psip, mup, 
                    ci = True, parpot = parpot_POT, varcov = varcov_POT)
            
            N, C, W = mev_fit(df1, ws = 1, how = 'pwm', threshold = 0)
            x0 = np.mean(XI)
            mq[ii], mqu[ii], mql[ii] = mev_CI(df1, 1-1/Tr, x0, ws = 1, 
                     ntimes = ntimes, MEV_how = 'pwm', MEV_thresh = 0.0, 
                                                        std_how = 'boot')        
        elif ci == False:
            parhat_GEV = gev.gev_fit(XI, how = 'lmom')
            csi, psi, mu = parhat_GEV
            gq[ii] = gev.gev_quant(1-1/Tr, csi, psi, mu)
            
            parhat_POT= gev.pot_fit(df1, datatype = 'df', way = 'ea', ea = 3, how = 'ml')
            csip, psip, mup = parhat_POT
            pq[ii] = gev.pot_quant(1-1/Tr, csip, psip, mup)
            
            N, C, W = mev_fit(df1, ws = 1, how = 'pwm', threshold = 0)
            x0 = np.mean(XI)
            mq[ii] = mev_quant(1-1/Tr,x0,N,C,W)
            
    if ci == True:        
        fig1 = plt.figure() # plot gev only
        ax1 = fig1.add_subplot(211)
        mytitle = 'Sliding window analysis, n='+str(winsize)+', Tr = '+str(Tr)+' years'
        # mytitle = 'Milano'
        ax1.set(title=mytitle, ylabel='Return level [mm]')
        ax1.plot(central_year, gq, color='red', label = 'GEV')
        # ax1.plot(central_year, pq, color='green', label = 'POT')
        ax1.fill_between(central_year, pql, pqu,
        alpha = 0.5, edgecolor='red', facecolor='red')
        ax1.legend(loc='upper left')
        ax2 = fig1.add_subplot(212, sharex = ax1)
        ax2.set( ylabel='Return level [mm]', xlabel='year')
        ax2.plot(central_year, mq, color='blue', label = 'MEV')
        ax2.fill_between(central_year, mql, mqu,
        alpha = 0.5, edgecolor='blue', facecolor='blue')
        ax2.legend(loc='upper left')                 
#        ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
        
#        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
        
        plt.show()
        
        fig2 = plt.figure() # plot MEV and GEV on the same plot
        ax = fig2.add_subplot(111)
        mytitle = 'Sliding window analysis, n='+str(winsize)+', Tr = '+str(Tr)+' years'
        # mytitle = 'Milano'
        ax.set(title=mytitle,
               ylabel='Return level [mm]', xlabel='year')
        ax.plot(central_year, mq, color='blue', label = 'MEV')
        ax.plot(central_year, gq, color='red', label = 'GEV')
        ax.fill_between(central_year, pql, pqu,
        alpha = 0.5, edgecolor='red', facecolor='red')
        ax.fill_between(central_year, mql, mqu,
        alpha = 0.5, edgecolor='blue', facecolor='blue')       
        ax.legend(loc='upper left')
        
#        ax.legend(loc=3,
#           ncol=2, mode="expand", borderaxespad=0.)
                
        if display == True:
            plt.show()
        
    elif ci == False: 
            
        fig1 = plt.figure() # plot MEV and GEV on the same plot
        ax = fig1.add_subplot(111)
        mytitle = 'Sliding window analysis, n='+str(winsize)+', Tr = '+str(Tr)+' years'
        ax.set(title=mytitle ,
               ylabel='Return level [mm]', xlabel='year')
        ax.plot(central_year, mq, color='blue', label = 'MEV')
        ax.plot(central_year, gq, color='red', label = 'GEV')
        ax.plot(central_year, pq, color='green', label = 'POT')        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if display == True:
            plt.show()
        
    if ci == True:
        return central_year, mq, mqu, mql, gq, gqu, gql, pq, pqu, pql, fig1, fig2
    else:
        return central_year, mq, gq, pq, fig1
    
    


