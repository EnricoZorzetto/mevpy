
''' 
###############################################################################

        Entico Zorzetto, 9/10/2017  
        enrico.zorzetto@duke.edu
        
        Set of functions to  calibrate and validate the MEV distribution
        most functions are to be applied to data frames 
        with the following fields:
            'PRCP' :: for the daily rainfall values
            'YEAR' :: for the observation year (in format yyyy)
            'DATE' :: date in format yyyymmdd
        years with just a few observation should not be used 
        (e.g., with more than 10% of missing values)
###############################################################################
'''
# import sys
import numpy as np
import pandas as pd
import scipy as sc
import mevpy.gev_fun as gev
from scipy.special import gamma
from scipy.stats import exponweib
import matplotlib.pyplot as plt

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
        if std_how = 'hess' hessian is used (onbly available for max like.)
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
    (when I do not know the asympt properties of the distr.)
    return std and optional pdf of fitted parameters  
    and their covariance matrix varcov
    fit to a sample of a distribution using the fitting function fittinfun
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
    only observations larger than a value 'threshold' are used for the fit
    -- threshold without renormalization -- it assumes the values below are 
    non existent. Default threshold = 0
    
    INPUT:: sample (array with observations)
           threshold (default is = 0)
    OUTPUT::
    returns numerosity of the sample (n) (only values above threshold)
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
    # do my calculations
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
    '''
    fit Weibull by means of Maximum_Likelihood _Estimator (MLE)
    finding numerically the max of the likelihood function
    return n size of sample >0 (used for fit)
    if std = True compute standard deviations and covariances 
    of parameters C and w.
    '''    
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
    # not sure - check this function -
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
    
def mev_fun(y,pr,N,C,W): 
    ''' MEV distribution function, to minimize numerically 
    for computing quantiles'''
    nyears = N.size
    mev0f = np.sum( ( 1-np.exp(-(y/C)**W ))**N  ) - nyears*pr 
    return mev0f


def mev_quant(Fi,x0,N,C,W): 
    '''  computes MEV quantiles for given non exceedance probabailities Fi'''
    Fi = np.asarray(Fi)
    is_scalar = False if Fi.ndim > 0 else True  
    Fi.shape = (1,)*(1-Fi.ndim) + Fi.shape    
    m = np.size(Fi)
    quant = np.zeros(m)
    for ii in range(m):
        myfun     = lambda y: mev_fun(y,Fi[ii],N,C,W)
        res       = sc.optimize.fsolve(myfun, x0,full_output = 1)
        quant[ii] = res[0]
        info      = res[1]
        fval      = info['fvec']
        if fval > 1e-5:
            print('mevd_quant:: ERROR - fsolve does not work -  change x0')
        quant  = quant if not is_scalar else quant[0]
    return quant


def mev_cdf(quant,N,C,W): 
    '''  computes mev cdf for given quantiles quant 
    given arrays of yearly parameters N,C,W'''
    quant       = np.asarray(quant)
    is_scalar   = False if quant.ndim > 0 else True
    quant.shape = (1,)*(1-quant.ndim) + quant.shape      
    nyears      = N.shape[0]
    m = np.size(quant)
    mev_cdf = np.zeros(m)
    for ii in range (m):
        mev_cdf[ii]     = np.sum( ( 1 - np.exp(-(quant[ii]/C)**W ))**N ) / nyears 
    mev_cdf     =  mev_cdf  if not is_scalar else  mev_cdf[0]
    return mev_cdf


def mev_fit(df, ws = 1, how = 'pwm', threshold = 0):
    ''' fit MEV to a dataframe of daily rainfall df - with PRCP, YEAR
    fitting Weibull to windows of size ws (scalar integer value, default is 1)
    
    how = fitting method. available 'ml', 'pwm', 'ls'
    default is pwm
    return arrays of Weibull parameters N,C,W
    arrays nwinsizes * nyears
    '''
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
        temp, Ci[iiw], Wi[iiw] = wei_fit(sample, how = how, threshold = threshold)        
    N = np.zeros(ncal2)
    for iiw in range(ncal2):
        sample = datamat_cal_2[iiw,:]
        wets = sample[sample > 0]
        N[iiw]=np.size(wets)
        
    C = np.repeat(Ci, winsize)
    W = np.repeat(Wi, winsize)
    return N,C,W


def mev_CI(df, Fi_val, x0, ws = 1, ntimes = 1000, MEV_how = 'pwm', 
                                         MEV_thresh = 0.0, std_how = 'boot'):
    '''non parametric bootstrap technique for MEV
    given a sample, at every iteration generates a new sample with replacement 
    and then fit again mev and obtain a quantiles 
    from data frame df ' missing data are assumed already treated
    by default it returns MEV 95% CI in hyp normal distr 
    USE NTIMES >> 20
    
    POSSIBLE OPTIONS FOR COMPUTING CONFIDENCE INTERVALS:
    ---------------------------------------------------------------------------
    CI_how = 'delta': use delta method under the hyp. that all are indep parameters
                --> and compute their individual affects on GEV quantiles
    CI_how = 'boot': do non parametric bootstrapping for daily values 
                          and number of events/year
                          NB: This might reduce variability and thus MEV quantiles?
                          then hyp. normal distr of quantiles
                          (DEFAULT)
    CI_how = 'boot_cdf': as before, but without normality assumption. But I need 
                         ntimes large enough to compute prob of 95% and 5%.
    CI_how = 'par': only resample from the arrays of (Ni, Ci, Wi) for each year/window - 
                --> am I missing some of the variability in this way?
                (default is boot)
    ---------------------------------------------------------------------------
    '''
    # perhaps reshuffling N as well would be better
    # I.e., parametric bootstrap - fix it
    
    # Write - case of asymmetric CI - with percentage instead of stdv
    # and one using the hessian and the delta method if possible
    Fi_val       = np.asarray(Fi_val)
    is_scalar   = False if Fi_val.ndim > 0 else True
    Fi_val.shape = (1,)*(1-Fi_val.ndim) + Fi_val.shape
    m = np.size(Fi_val) 
    QM = np.zeros((ntimes, m))
    
    N, C, W = mev_fit(df, ws = ws, how = MEV_how, threshold = MEV_thresh)
    Q_est = mev_quant(Fi_val, x0, N, C, W)

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
            N, C, W = mev_fit(dfr, ws, how = MEV_how, threshold = MEV_thresh)
            QM[ii,:] = mev_quant(Fi_val, x0, N, C, W)
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
            N, C, W = mev_fit(dfr, ws, how = MEV_how, threshold = MEV_thresh)
            QM[ii,:] = mev_quant(Fi_val, x0, N, C, W)
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
    return Q_est, Q_up, Q_low
        
        
#def mev_boot(df):
#    ''' non parametric bootstrap technique for MEV
#    reshuffle i) the number of events for each year in the series
#    ii) the daily events. For each year generates N_i events.
#    For both steps, we sample with replacement.'''
#    # check this function - 
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
    # check this function - 
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

def remove_missing_years(df,nmin):
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
    return df,nyears2, nyears1


def tab_rain_max(df):
    '''  input has to be a pandas data frame df
    including the variables YEAR, PRCP
    return vectors of annual maxima (ranked ascend), emp cdf, return time
    Default using Weibull plotting position for non exceedance probability'''
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


def table_rainfall_maxima(df, how = 'pwm', thresh = 0):
    '''
    input has to be a pandas data frame df
    including the variables YEAR, PRCP
    and a threshold for fitting Weibull parameters
    return vectors of annual maxima (ranked ascend), emp cdf, return time
    Default using Weibull plotting position for non exceedance probability
    and weibull NCW for each year in the record
    '''
    years_all  = df['YEAR']
    years      = pd.Series.unique(years_all)
    nyears     = len(years)
    maxima     = np.zeros([nyears,1])
    NCW        = np.zeros([nyears,3])
    for jj in range(nyears):
        my_year      = df.PRCP[df['YEAR'] == years[jj]]
        maxima[jj,0] = np.max(my_year)
        (NCW[jj,0], NCW[jj,1], NCW[jj,2]) = wei_fit(my_year , how = how, 
                                                       threshold = thresh)
    XI = np.sort(maxima,axis = 0) # default ascend
    Fi = np.arange(1,nyears+1)/(nyears + 1)
    TR = 1/(1 - Fi)  
    return XI,Fi,TR,NCW


def fit_EV_models(df, tr_min = 5, ws = 1, GEV_how = 'lmom', MEV_how = 'pwm', 
                 MEV_thresh = 0, POT_way = 'ea', POT_val = 3, POT_how = 'ml',
        ci = False, ntimes = 1000, std_how_MEV = 'boot', std_how_GEV = 'hess', 
                                            std_how_POT = 'hess', rmy = 36):
    ''' fit MEV, GEV and POT to daily data in the dataframe df
    with fields PRCP and YEAR, and compare them with original annual maxima
    compute quantiles - and non exceedance probabilities
    and compare with the same dataset / produce QQ and PP plots
    default methods are PWM, LMOM, and ML for MEV-GEV-POT respectively
    MEV - fit Weibull to windows of size ws, default ws = 1 (yearly Weibull)'''
    # ADD COMPUTATION OF CONFIDENCE INTERVALS
    df, ny2, ny1 = remove_missing_years(df,rmy)
    XI,Fi,TR     = tab_rain_max(df)
    tr_mask      = TR > tr_min
    TR_val       = TR[tr_mask]
    XI_val       = XI[tr_mask]
    Fi_val       = Fi[tr_mask]
    #x0           = np.mean(XI_val) - 0.2*np.std(XI_val)
    
    x0 = 50.0
    # MOD - try a few different
    # print(x0)
    # fit distributions
    csi,psi,mu      = gev.gev_fit(XI, how = GEV_how)
    N, C, W         = mev_fit(df, ws= ws, how = MEV_how, threshold = MEV_thresh)
    csip, psip, mup = gev.pot_fit(df, datatype = 'df', way = POT_way, ea = POT_val,
                               sp = POT_val, thresh = POT_val,  how = POT_how)
    # compute quantiles
    QM           = mev_quant(Fi_val, x0, N, C, W)
    QG           = mu + psi/csi*(( -np.log(Fi_val))**(-csi) -1)
    QP           = mup + psip/csip*(( -np.log(Fi_val))**(-csip) -1)
    # compute non exceedance frequencies
    FhM          = mev_cdf(XI_val,N,C,W)
    FhG          = gev.gev_cdf(XI_val, csi, psi, mu)
    FhP          = gev.gev_cdf(XI_val, csip, psip, mup)
    
    if ci:
        # MEV - re-evaluate the mean here!
        QmM, QuM, QlM = mev_CI(df, Fi_val, x0, ws = ws, ntimes = ntimes, 
                  MEV_how = MEV_how, MEV_thresh = MEV_thresh , std_how = std_how_MEV)        
        # POT
        parhat_POT, parpot_POT, parstd_POT, varcov_POT = gev.pot_fit(df, datatype = 'df', way = POT_way, ea = POT_val, 
                  sp = POT_val, thresh = POT_val, how = POT_how, std = True, std_how = std_how_POT, std_num = ntimes)
        QmP, QuP, QlP = gev.pot_quant(Fi_val, csip, psip, mup, ci = True, parpot = parpot_POT, varcov = varcov_POT)
        # GEV
        parhat_GEV, parstd_GEV, varcov_GEV = gev.gev_fit(XI, how = GEV_how,
                        std = True, std_how = std_how_GEV, std_num = ntimes)
        QmG, QuG, QlG = gev.gev_quant(Fi_val, csi, psi, mu, ci = True, varcov = varcov_GEV)
        
        return TR_val, XI_val, Fi_val, QM, QG, QP, QuM, QuG, QuP, QlM, QlG, QlP, FhM, FhG, FhP 
        # return TR_val, XI_val, Fi_val, QM, QG, QP, QmM, QmG, QmP, QuM, QuG, QuP, QlM, QlG, QlP, FhM, FhG, FhP 

    else:
        return TR_val, XI_val, Fi_val, QM, QG, QP, FhM, FhG, FhP
    

def shuffle_mat(datamat, nyears, ncal, nval):
    # this only shuffles YEARS in a sample of daily rainfall -
    # for cross validation of extreme value models
    '''given an array with shape (nyears*ndays)
    scramble its years
    and returns a calibration martrix, calibration maxima,
    and independent validation maxima'''   
    randy     = np.random.permutation( int(nyears) ) 
    datamat_r = datamat[randy]
    mat_cal   = datamat_r[:ncal, :]
    maxima    = np.max(datamat_r, axis = 1) # yearly maxima
    max_cal   = maxima[:ncal]
    max_val   = maxima[ncal:ncal+nval]
    return mat_cal, max_cal, max_val, datamat_r


def shuffle_all(datamat, nyears, ncal, nval):
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
        
        
        
        
        
        
        
#    randy     = np.random.permutation( int(nyears) ) 
#    datamat_r = datamat[randy]
#    mat_cal   = datamat_r[:ncal, :]
#    maxima    = np.max(datamat_r, axis = 1) # yearly maxima
#    max_cal   = maxima[:ncal]
#    max_val   = maxima[ncal:ncal+nval]
#    return mat_cal, max_cal, max_val, datamat_r


def cross_validation(df, ngen, ncal, nval, tr_min = 5, ws=[1],
                     GEV_how = 'lmom', MEV_how = 'pwm', MEV_thresh = 0,
                 POT_way = 'ea', POT_val = [3], cross = True, ncal_auto = 100,
                 shuff = 'year'):
    ''' FIT MEV and GEV and perform validation with stationary time series
    obtained reshuffling the years of the original time series    
    ###########################################################################
    INPUT::
        df -  dataframe with fields 'PRCP' daily precipitation values (float)
                                    'YEAR' year in format yyyy (integer/float)
        ngen - number of random reshufflings of the dataset
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
    
    em        = np.zeros((ngen, nwinsizes, ntr))
    eg        = np.zeros((ngen, ntr))
    ep        = np.zeros((ngen, nthresh, ntr))
    m_rmse    = np.zeros((nwinsizes, ntr))
    g_rmse    = np.zeros(ntr)
    p_rmse    = np.zeros((nthresh, ntr))
     
    for iig in range(ngen): # loop on random generations
        
        if shuff == 'year': # resample years only, or
            mat_cal, max_cal, max_val, datamat_r = shuffle_mat(datamat, nyears, 
                                                              ncal, nval)
        elif shuff == 'all': # or reshuffle daily values
            mat_cal, max_cal, max_val, datamat_r = shuffle_all(datamat, nyears, 
                                                              ncal, nval)
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
                temp, Ci[iiw], Wi[iiw]  = wei_fit(sample , how = MEV_how, 
                                                       threshold = MEV_thresh)                
            N = np.zeros(ncal2)
            for iiw in range(ncal2):
                sample = datamat_cal_2[iiw,:]
                wets = sample[sample > MEV_thresh]
                N[iiw]=np.size(wets)
                
            C = np.repeat(Ci, winsize)
            W = np.repeat(Wi, winsize)
            QM  = mev_quant(Fi_val, x0, N, C, W)
            for iitr in range(ntr): # comput emev relative errors
                em[iig, iiws, iitr] = (QM[iitr] - XI_val[iitr])/XI_val[iitr]
                
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
    return TR_val, m_rmse, g_rmse, p_rmse, em, eg, ep


def slideover(df, winsize = 30, Tr = 100, display = True, ci = True, ntimes = 100):
    ''' perform EV analysis on sliding and overlapping windows'''
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
    
    
###############################################################################
###############################################################################
    
##############   FUNCTIONS TO BE REMOVED IN A FUTURE VERSION - SLOW ###########

###############################################################################
###############################################################################
    
def shuffle_years(df):
    # This function is awfully slow -is not necessary, use shuffle_mat instead!
    ''' given a dataframe df with YEAR, PRCP columns
    produce a dataframe with observed years in reshuffled order'''
    years = np.unique( df.YEAR )
    neworder = list( np.random.permutation(years) )
    dfm = df.set_index(['YEAR', 'DATE'], inplace = False) # multiindex object
    newindex = sorted(dfm.index, key = lambda x: neworder.index(x[0]))
    dfm2 = dfm.reindex(newindex)
    dfR = dfm2.reset_index(inplace = False) 
    ###########################################################################
    ############ Tested  12-10-2017 with the following df #####################
    ###########################################################################
    #A = np.array( [[1990, 1990, 1990, 1991, 1991, 1991, 1992, 1992, 1992],
    #                [19900101, 19900102, 19900103, 19910101, 19910102, 
    #                            19910103, 19920101, 19920102, 19920103],
    #                [1,2,3,4,5,6,7,8,9],
    #                [11,12,13,14,15,16,17,18,19]]).transpose()    
    #B = pd.DataFrame(A)    
    #B.columns = ['YEAR', 'DATE', 'PRCP', 'OTHER']
    #dfR = mev.shuffle_years(B)
    ###########################################################################
    ###########################################################################
    return dfR

# molto lento
def mevd_quant_windows(df, Fi, x0, winsize = 1, how = 'pwm', threshold = 0):
    ''' returns an array of MEV-estimated quantiles QM
    fitting MEV to the daily rainfall data in the dataframe df, 
    which must contain entries YEAR and PRCP.
    x0 = initial guess for numerical procedure
    winsize = size [in years] of the window for fitting Weibull -default 1 year
    how = fitting method for Weibull -default is probability weighted moments
    (other choices are 'ml' -for max.likelihood and  'ls' -for least seuares)
    threshold = for fitting Weibull. default value is zero.
    Only available for pwm
    '''
    # very slow - use cross validation instead
    # print(df.head())
    years = np.unique( df.YEAR)
    # print(years)
    ncal =  np.size( years ) # number of years of calibraiton
    # print(ncal)
    # print(winsize)
    # winsize = 5
    # numwin = np.int8( np.floor(ncal/winsize))
    numwin = ncal//winsize # integer part of the division
    ncal2 = numwin*winsize
    # print(numwin)
    # print(winsize)
    # print(ncal2)
    years2 = years[:ncal2]
    to_exclude = years[ncal2:]
    df2 = df[~df.YEAR.isin(to_exclude)]

    all_groups = np.repeat( np.arange(numwin), winsize)    
    equiv = dict(zip(years2, all_groups)) # dictionary
    # df2["GROUP"] = df2["YEAR"].map(equiv)
    df2["GROUP"] = df2["YEAR"].map(equiv)
    # df2['GROUP'] = pd.Series(np.zeros(df2.size))
    # df2.loc['GROUP'] = df2['YEAR'].map(equiv)
    groups = np.unique(all_groups)
    
    # fit N for each year
    N = np.zeros(ncal2)
    for ii in range(ncal2):
        dataii = df2.PRCP[df2.YEAR == years2[ii]]
        N[ii] = np.size( dataii[dataii > 0] )
        
    # fit C and w for each group
    Ci = np.zeros(numwin)
    Wi = np.zeros(numwin)    
    for ii in range(numwin):
        sample = df2.PRCP[df2.GROUP == groups[ii]]
        # fit Weibull in the selected fashion:
        ( ni, Ci[ii], Wi[ii] )   = wei_fit(sample , how = how, 
                                                       threshold = threshold)                
        ni, Ci[ii], Wi[ii] = wei_fit_pwm(sample)
    C = np.repeat(Ci, winsize)
    W = np.repeat(Wi, winsize)
    
    # compute MEV quantiles for the given NCW
    QM = mev_quant(Fi, x0, N, C, W) 
    return QM

# function probably not very useful anymore
# to be removed in future versions
def mevd_qufit(df, Fi, x0, how = 'pwm', threshold = 0): 
    # version to be removed in the future
    '''
    fit MEV to a data frame sample
    and compute quantiles for a range of non exceedance probab Fi
    computes the MEV quantile for a given return time
    '''
    years_all  = df['YEAR']
    years      = pd.Series.unique(years_all)
    nyears     = len(years)
    NCW        = np.zeros([nyears,3])
    for jj in range(nyears):
        my_year      = df.PRCP[df['YEAR'] == years[jj]]
        (NCW[jj,0], NCW[jj,1], NCW[jj,2]) = wei_fit(my_year, how = 'pwm', 
                                                       threshold = threshold)              
    m = np.size(Fi)
    QM = np.zeros(m)
    for ii in range(m):
        myfun = lambda y: mev_fun(y,Fi[ii],NCW[:,0], NCW[:,1], NCW[:,2])
        res   = sc.optimize.fsolve(myfun, x0,full_output = 1)
        QM[ii] = res[0]
        info = res[1]
        fval = info['fvec']
        # print(fval)
        if fval > 1e-5:
            print('warning - there is something wrong solving fsolve')
    return QM

