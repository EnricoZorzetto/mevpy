
''' 
###############################################################################

        Entico Zorzetto, 16/10/2017  
        enrico.zorzetto@duke.edu
        
        Set of functions to  calibrate and validate the GEV distribution
        and POT method
        most functions are to be applied to data frames 
        with the following fields:
            'PRCP' :: for the daily rainfall values
            'YEAR' :: for the observation year (in format yyyy)
            'DATE' :: date in format yyyymmdd
        years with just a few observation should not be used 
        (e.g., with more than 10% of missing values)
###############################################################################
'''
# add
# confidence intervals
# declustering
# threshold selection functions

import sys
import numpy as np
from scipy.special import gamma
import scipy as sc
from numpy.linalg import inv

###############################################################################

############################ GEV BASIC FUNCTIONS ##############################

###############################################################################


def gev_fit(sample, how = 'lmom', std = False, std_how = 'hess', std_num = 1000):
    ''' fit GEV with one of the following methods:
        how = 'lmom' for probability weighted moments
        how = 'ml' for maximum likelihood
        (default is lmom) 
        returns: parhat = (csi, psi, mu)'
        optional: if std = True (default is false)
        compute parameter est. standard deviations. parstd
        and their covariance matrix varcov
        if std_how = 'boot' bootstrap is used 
        if std_how = 'hess' hessian is used (onbly available for max like.)
        std_num --> number or resamplings in the bootstrap procedure.
        default is 1000. '''
    if   how == 'lmom':
        parhat = gev_fit_lmom(sample)
    elif how == 'ml' and std == True and std_how == 'hess':
        parhat, parstd, varcov  = gev_fit_ml(sample, std = True)
    elif how == 'ml':
        parhat  = gev_fit_ml(sample)
    else:
        print(' ERROR - insert a valid fitting method for GEV ')
    if std == True:
        if how == 'lmom':
            parstd, varcov = bootstrap(sample, fitfun = gev_fit_lmom, npar= 3, ntimes = std_num)
        elif how == 'ml' and std_how == 'boot':
            parstd, varcov = bootstrap(sample, fitfun = gev_fit_ml, npar = 3, ntimes = std_num)
        return parhat, parstd, varcov
    else:
        return parhat 


def gev_fit_lmom(sample):
    ''' Fit GEV distribution to a sample of annual maxima
    by means of LMOM technique (Hosking 1990 &co0)
    maxima must be numpy column array
    return parhat = (csi, psi, mu) i.e., (GEV shape scale location)
    rem here csi > 0 --> Heavy tailed distribution'''
    sample  = np.asarray(sample)
    n       = np.size(sample)
    x       = np.sort(sample, axis = 0)
    b0      = np.sum(x)/n
    b1      = 0.0
    for j in range(0,n): # skip first element
        jj   = j + 1 # real index
        b1   = b1 + (jj - 1)/(n - 1)*x[j]
    b1   = b1/n
    b2   = 0.0
    for j in range(0,n): # skip first two elements
        jj  = j + 1 # real
        b2  = b2 + (jj-1)*(jj-2)/(n-1)/(n-2)*x[j]
    b2   = b2/n
    # L MOMENTS - linear combinations of PWMs
    L1   = b0
    L2   = 2*b1 - b0
    L3   = 6*b2-6*b1+b0
    t3   = L3/L2  #L skewness    
    # GEV parameters from L moments ( Hoskins 1990)
    # using Hoskins (1985) approximation for computing k:
    c   = 2/(3 + t3) - np.log(2)/np.log(3)   
    k   = 7.8590*c + 2.9554*c**2
    csi = -k                                      # ususal shape 
    psi = L2*k/((1 - 2**(-k))*gamma(1 + k))       # scale 
    mu  = L1 - psi*(1 - gamma(1 + k))/k            # location
    parhat = (csi, psi, mu)
    return parhat
    

def sample_lmom(sample):
    '''
    return L1, L2, t3
    the first two L-mom and the L moment ratio t3 = L3/L2
    - this is consistent with Hosking's module -checked
    '''
    sample = np.asarray(sample)
    n      = np.size(sample)
    x      = np.sort(sample,axis=0)
    b0     = np.sum(x)/n
    b1     = 0.0
    for j in range(0,n): # skip first element
        jj   = j+1 # real
        b1   = b1+(jj-1)/(n-1)*x[j]
    b1   = b1/n
    
    b2   = 0.0
    for j in range(0,n): # skip first two elements
        jj  = j+1 # real
        b2  = b2+(jj-1)*(jj-2)/(n-1)/(n-2)*x[j]
    b2   = b2/n
    # L MOMENTS - linear combinations of PWMs
    L1  = b0
    L2  = 2*b1-b0
    L3  = 6*b2-6*b1+b0
    t3  = L3/L2  #L skewness
    return L1, L2, t3
    
    
def gev_quant(Fi, csi, psi, mu, ci = False, varcov = []):
    ''' compute GEV quantile q for given non exceedance probabilities in Fi
    with parameters csi, psi, mu (shape, scale, location) 
    optional: if ci = True also produce the upper and lower confidence 
    intervals obtained under the hyp of normal distribution.
    In this case the covariance matrix of the parameters must be provided
    varcov = variance-covariance matrix of parameters.''' 
    Fi         = np.asarray(Fi)
    is_scalar  = False if Fi.ndim > 0 else True
    Fi.shape   = (1,)*(1-Fi.ndim) + Fi.shape
    q          = mu+psi/csi*((-np.log(Fi))**(-csi)-1)
    if ci == True:
        # compute std of quantiles using the DELTA METHOD
        m = np.size(Fi)
        qu = np.zeros(m)
        ql = np.zeros(m)
        for ii in range(m):
            yr = -np.log(Fi[ii])
            DEL = np.array([psi*csi**(-2)*(1-yr**(-csi))-psi*csi**(-1)*yr**(-csi)*np.log(yr),
                   -csi**(-1)*(1-yr**(-csi)), 1])
            prod1 = np.dot(varcov, DEL)
            varz = np.dot( prod1, DEL)    
            stdz = np.sqrt(varz)
            ql[ii] = q[ii] - 1.96*stdz
            qu[ii] = q[ii] + 1.96*stdz            
        qu = qu if not is_scalar else  qu[0]
        ql = ql if not is_scalar else  ql[0]
        q  =  q if not is_scalar else  q[0]
        return q, qu, ql
    else:
        q  =  q if not is_scalar else  q[0]
        return q
  
                         
def gev_cdf(q, csi, psi, mu):
    ''' compute GEV non exceedance probabilities Fhi for given quantiles in q
    with parameters csi, psi, mu (shape scale location) '''  
    q          = np.asarray(q)
    is_scalar  = False if q.ndim > 0 else True
    q.shape    = (1,)*(1-q.ndim) + q.shape  
    Fhi        = np.exp( -(1 + csi/psi*(q-mu))**(-1/csi))
    Fhi        =  Fhi if not is_scalar else  Fhi[0]
    return Fhi 
     
               
def gev_random_quant(length, csi, psi, mu):
    ''' generates quantiles randomly extracted from a GEV distribution 
    with parameters shape (csi > 0 if Frechet), scale (psi), and location (mu)
    returns an array of size length or a scalar when length = 1 '''
    Fi = np.random.rand(length)
    if csi != 0.0:
        xi = mu+psi/csi*((-np.log(Fi))**(-csi)-1) # Frechet/inverse Weibull
    else:
        xi = mu - psi * np.log ( -np.log(Fi)) # Gumbel
    xi    = xi if length > 1 else xi[0]
    return xi


def bootstrap(sample, fitfun, npar = 3, ntimes = 1000):
    '''non parametric bootstrap technique 
    for computing confidence interval for a distribution
    (when I do not know the asympt properties of the distr.)
    return std and optional pdf of fitted parameters  
    and their covariance matrix varcov
    fit to a sample of a distribution using the fitting function fittinfun
    with a number of parameters npar (default is 3 for GEV)'''
    n = np.size(sample)
    # resample from the data with replacement
    parhats = np.zeros((ntimes,npar))
    for ii in range(ntimes):
        replaced = np.random.choice(sample,n)
        parhats[ii,:] = fitfun(replaced)   
    parstd = np.std(parhats, axis = 0)
    varcov = np.cov(parhats, rowvar = False)
    return parstd, varcov


def hess(fun, y, data):
    ''' numeric Hessian matrix
    for estimating MLE parameters confidence intervals'''
    ep = 0.0001
    x = np.array(y)
    eps = ep*x
    n = np.size(x)
    m = np.zeros((n,n))
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    x3 = np.zeros(n)
    x4 = np.zeros(n)
    for i in range(n):
        for j in range(n):
            x1[:] = x[:]
            # I modify the original array as well - it is a view!
            x1[i] = x1[i] + eps[i]
            x1[j] = x1[j] + eps[j]
            x2[:] = x[:]
            x2[i] = x2[i] + eps[i]
            x2[j] = x2[j] - eps[j]
            x3[:] = x[:]
            x3[i] = x3[i] - eps[i]
            x3[j] = x3[j] + eps[j]
            x4[:] = x[:]
            x4[i] = x4[i] - eps[i]
            x4[j] = x4[j] - eps[j]
            m[i,j] = (fun(x1, data) -fun(x2, data) - fun(x3, data) + fun(x4, data))/(4*eps[i]*eps[j]) 
    M = np.asmatrix(m)
    return inv(M)
    

def gev_fit_ml(data, std = False):
    ''' fit GEV by means of ML estimation to an array of observations (data)
    e.g., of annual maxima
    return GEV parameters -->  parhat = (csi, psi, mu)
    shape csi (>0 Frechet), scale psi, and location mu'
    and if stdv = True (optional) it does also return their stdv
    parstd = (std_csi, std_psi, std_mu)
    and their covariance matrix'''    
    # first guess with method of moments
    sigma0 = np.sqrt((6*np.var(data))/np.pi)
    mu0    = np.mean(data) - 0.57722 * sigma0
    xi0    = 0.1
    theta  =[xi0,sigma0,mu0]
    
    y = 1 + (theta[0] * (data - theta[2]))/theta[1]
    if theta[1] < 0.0 or np.min(y) < 0.0:
        theta[1] = 1
        theta[2] = 1
    
    ###########################################################################
    # neg log likelihood function          
    def negloglikegev(theta, data):
        y = 1 + (theta[0] * (data - theta[2]))/theta[1]
        if theta[1] < 0.0 or np.min(y) < 0.0:
            	c= np.inf
        else: 
            term1 = len(data) * np.log(theta[1])
            term2 = np.sum((1 + 1/theta[0]) * np.log(y))
            term3 = np.sum(y**(-1/theta[0]))
            c = term1 + term2 + term3
        return c        
    ###########################################################################   
    
    nllg = lambda theta: negloglikegev(theta, data)
    res = sc.optimize.minimize(nllg, theta, method='nelder-mead', 
                                        options={'xtol': 1e-8, 'disp': False})
    
    parhat = res.x # MLE PARAMETERS
    # csi = res.x[0] # GEV shape (>0 Frechet)
    # psi = res.x[1] # GEV scale
    # mu = res.x[2] # GEV shape location
    
    # OPTIONAL - COVARIANCE OF ESTIMATED PARAMETERS
    if std:
        varcov = hess(negloglikegev, parhat, data)
        parstd = np.sqrt( np.diag(varcov) )
        return parhat, parstd, varcov
    else:
        return parhat


def pot_fit(data, datatype = 'df', way = 'ea', ea = 3, sp = 0.1, thresh = 10, 
                how = 'ml', std = False, std_how = 'hess', std_num = 1000):
    '''
    -----------------------------------------------------------------
    fit GEV by means of Peak Over Threshold (POT) method
    to data in a data frame df with fields PRCP, YEAR
    select a threshold in one way
    1) way = 'thresh' -> specify the threshold
    2) way = 'ea' -> fixed average number of exceedances / year
    3) way = 'sp' -> fixed survival probability of being above threshold
    when one is chosed the other inputs (among ea, sp, thresh) are ignored
    
    RETURNS: parhat = (csi, psi, mu)
    -----------------------------------------------------------------         
    INPUTS:  data: can be data frame with fields YEAR, PRCP
                   or array of shape ( nyears * ndays )
             data_type = 'df' if data is a dataframe df
                       = 'mat' if it is an array of data ->dim:  nyears*ndays
                       (default is data frame)
             way
             ea = 5 average excesses / year
             sp = 0.1 survival probability
             thresh = 10 
             how: GPD fitting method. 
                 can be 'pwm' or 'ml'. 
                 default is 'ml'
             nblocks (te.g., nyears of data: to compute arrival rate)
    ------------------------------------------------------------------
    alternatively, specify one of the following:
    - ea = average number of exceedances / year
    - sp = thresh survival probablibility (p of being above)
    '''
    if datatype == 'df':
        sample = data.PRCP
        years = np.unique(data.YEAR)
        nyears = np.size(years)
    elif datatype == 'mat':
        sample = np.ravel(data) # flatten array
        nyears = data.shape[0]
    else:
        print('ERROR :: insert a valid data type')
    
    wets = sample[sample > 0]
    num  = np.size(wets)
    xi =  -np.sort(-wets) # to sort descend
    fi = (np.arange(num) + 1)/(num + 1) # survival probability if descend
    
    # threshold selection
    if way == 'sp':
        th_index = np.abs( fi - sp ).argmin()
        threshold = xi[th_index]
    elif way == 'ea':
        ne = np.rint(ea*nyears).astype(np.int32) # number of elements above it
        threshold = xi[ne]
    elif way == 'thresh':
        threshold = thresh
    else:
        print('ERROR :: insert a valid threshold selection method')        
    # compute exceedances over threshold
    exceedances   = sample[sample > threshold]
    excesses      = exceedances - threshold    
    nexc          = np.size(excesses)
    poi_lambda    = nexc/nyears # events/block
    # while this is the variance of an arrival rate (hyp Binomial)
    var_lambda    = poi_lambda/365.25*(1-poi_lambda/365.25)/np.size(sample)
    # Fit GPD
    if std == True and std_how == 'hess':
        pargpd, stdgpd, covgpd = gpd_fit(excesses, how = how, std = True, std_how = std_how, std_num = std_num)
    else:
        pargpd = gpd_fit(excesses, how = how)
    xi = pargpd[0]
    beta = pargpd[1]    
    # GEV parameters based on GPD/Poisson par
    csi           = xi
    psi           = beta*poi_lambda**xi
    mu            = threshold - beta/xi*(1-poi_lambda**xi)
    parpot = xi, beta, poi_lambda
    parhat = csi, psi, mu
    if std == True:
        varcov = np.zeros((3,3))
        # here lambda as a rate, not number of arrivals/year
        if std_how == 'hess':
            varcov[0,0] = var_lambda  # variance of lambda arrival rate
            varcov[1:3, 1:3] = covgpd
            parstd = np.sqrt(np.diag(varcov))
        elif std_how == 'boot':
            parstd, varcov = pot_boot(excesses, var_lambda, GPD_how = how, ntimes = std_num)
        else:
            print('pot_fit ERROR: specify a valid std_how method')
            
        return parhat, parpot, parstd, varcov
    else:
        return parhat


def pot_boot(sample, var_lambda, GPD_how = 'ml', ntimes = 1000):
    '''non parametric bootstrap technique 
    for GEV parameters obtained with POT method'''
    # resample from the data with replacement
    # only resample from the sample of the excesses
    # and use the theoretical variance for lambda
    varcov = np.zeros((3,3))
    parstd = np.zeros(3)
    n = np.size(sample)
    varcov[0,0] = var_lambda
    parstd[0] = np.sqrt(varcov[0,0])
    parhats = np.zeros((ntimes, 2))
    # compute variance of the GPD parameters
    for ii in range(ntimes):
        replaced = np.random.choice(sample,n)
        parhats[ii,:] = gpd_fit(replaced, how = GPD_how) 
    parstd[1:3] = np.std(parhats, axis = 0)
    varcov[1:3,1:3] = np.cov(parhats, rowvar = False)
    return parstd, varcov


def pot_quant(Fi, csi, psi, mu, ci = False, parpot = [], varcov = []):
    ''' POT - GEV quantiles and confidence intervals
    input: GEV parameter estimated with POT -csi, psi, mu
    POT parpot = (csi, beta, lambda) and their covariance matrix varcov
    but GPD + Poisson variances instead'''
    Fi         = np.asarray(Fi)
    is_scalar  = False if Fi.ndim > 0 else True
    Fi.shape   = (1,)*(1-Fi.ndim) + Fi.shape
    q          = mu+psi/csi*((-np.log(Fi))**(-csi)-1)
    
    if ci == True:
        # compute std of quantiles using the DELTA METHOD
        m = np.size(Fi)
        qu = np.zeros(m)
        ql = np.zeros(m)        
        beta = parpot[1]
        lamb = parpot[2]
        for ii in range(m):
            logyr = np.log( lamb/-np.log(Fi[ii]))
            yr = (lamb/-np.log(Fi[ii]))**csi
            # order lambda csi beta
            DEL = np.array([ beta/lamb*yr, 
                    -beta/csi**2*(yr-1) + beta/csi*yr*logyr,
                            1/csi*(yr-1)])
    
            prod1 = np.dot(varcov, DEL)
            varz = np.dot( prod1, DEL)    
            stdz = np.sqrt(varz)
            ql[ii] = q[ii] - 1.96*stdz
            qu[ii] = q[ii] + 1.96*stdz            
        qu = qu if not is_scalar else  qu[0]
        ql = ql if not is_scalar else  ql[0]
        q  =  q if not is_scalar else  q[0]
        return q, qu, ql
    else:
        q =  q if not is_scalar else  q[0]
        return q
    
    

###############################################################################
    
#################### GENERALIZED PARETO BASIC FUNCTIONS #######################
    
###############################################################################


def gpd_fit(sample, how = 'ml', std = False, std_how = 'hess', std_num = 1000):
    ''' fit a 2-parameters GPD with one of the following methods:
        how = 'pwm' for probability weighted moments
        how = 'ml' for maximum likelihood 
        (default is ml) 
        returns: parhat = (csi, beta) - shape and scale
        optional: if std = True (default is false)
        compute parameter est. standard deviations. parstd
        and their covariance matrix varcov
        if std_how = 'boot' a bootstrap technique is used 
        if std_how = 'hess' hessian is used (onbly available for max like.)
        std_num --> number or resamplings in the bootstrap procedure.
        default is 1000. '''
    if   how == 'pwm':
        parhat = gpd_fit_pwm(sample)
    elif how == 'ml' and std == True and std_how == 'hess':
        parhat, parstd, varcov  = gpd_fit_ml(sample, std = True)
    elif how == 'ml':
        parhat  = gpd_fit_ml(sample)
    else:
        print(' ERROR - insert a valid fitting method for GEV ')
    if std == True:
        if how == 'pwm':
            parstd, varcov = bootstrap(sample, gpd_fit_pwm, npar = 2, ntimes = std_num)
        elif how == 'ml' and std_how == 'boot':
            parstd, varcov = bootstrap(sample, gpd_fit_ml, npar = 2, ntimes = std_num)
        return parhat, parstd, varcov
    else:
        return parhat

def gpd_fit_ml(data, std = False):
    '''
    fit a 2 parameter GPD to a sample
    by means of maximum likelihood estimation
    RETURNS: csi (shape, >0 for heavy tailed case)
             beta (scale par)
             location mu is always zero
    WARNING: not sure the algorithm minimize the function well enough -
    I should provide jacobian and use Newton instead
    '''
    # first guess GPD parameters
    xbar = np.mean(data)
    s2   = np.var(data)
    xi0 = -0.5*(((xbar**2)/s2)-1)
    beta0 = 0.5*xbar*(((xbar**2)/s2)+1)
    theta=[xi0, beta0]
    
    xi    = theta[0]
    beta  = theta[1]
    cond1 = beta <= 0
    cond2 = xi <= 0 and np.max(data) > ( - beta/xi)
    if cond1 or cond2:
        theta[0] = 1
        theta[1] = 1
    
    ###########################################################################
    # neg log likelihood function  
        
    def negloglikegpd(theta, data):
        xi   = theta[0]
        beta = theta[1]
        cond1 = beta <= 0
        cond2 = xi <= 0 and np.max(data) > ( - beta/xi)
        if cond1 or cond2:
            f = np.inf
        else:
            y = np.log(1 + (xi * data)/beta)/xi
            # y/xi
            f = len(data) * np.log(beta) + (1 + xi) * np.sum(y)
        return f
    
    def jacobf(theta,data):
        '''
        jacobian - gradient - of the negloglikegpd function
        jac[0] = d/dxi
        jac[1] = d/dbeta
        '''
        jacobf = np.zeros(2)
        xi   = theta[0]
        beta = theta[1]
        cond1 = beta <= 0
        cond2 = xi <= 0 and np.max(data) > ( - beta/xi)
        if cond1 or cond2:
            jacobf[0] = np.inf
            jacobf[1] = np.inf
        else:
            y = data/beta /(1 + (xi * data)/beta)
            y2 = (1 + (xi * data)/beta)
            jacobf[0] = -1/xi**2 *np.sum(y2) + (1 + 1/xi) * np.sum(y)
            jacobf[1] = len(data)/beta + (1 + 1/xi) * np.sum( -xi/beta*y)
        return jacobf
    
    ###########################################################################

    # this algorithm seem to work fairly well with the jacobian
    likefun = lambda theta: negloglikegpd(theta, data)
    # jacob = lambda theta: jacobf(theta, data)
    
    # res2 = sc.optimize.minimize(likefun, theta, method='L-BFGS-B', jac=jacob,
    #                                    options={'gtol': 1e-8, 'disp': False})
    
    res2 = sc.optimize.minimize(likefun, theta, method='Nelder-Mead')
#     res2 = sc.optimize.minimize(likefun, theta, method='Powell')


        
    if not res2.success:
        print('gpd_ml_fit does not converge. ')
        print(res2)
        sys.exit()
    parhat = res2.x # csi, beta
    # csi = res2.x[0] # GPD shape (>0 Frechet)
    #beta = res2.x[1] # GPD scale
    
    # OPTIONAL - COVARIANCE OF ESTIMATED PARAMETERS

    if std:
        varcov = hess(negloglikegpd, parhat, data)
        # print('varcov = ')
        # print(varcov)
        parstd = np.sqrt( np.diag(varcov) )
        return parhat, parstd, varcov
    else:
     return parhat


def gpd_fit_pwm(sample, threshold = 0): 
    '''
    fit a 2-parameters GPD distribution to a sample 
    by means of Probability weighted moments matching 
    SEE Hosking and Wallis 1987
    'Parameter and Quantile Estimation for the Generalized Pareto Distribution'
    returns shape (csi >0 for heavy tailed distr)
    and scale par. beta
    OPTIONAL threshold to ignore smaller values - without renormalizing.
    '''
    sample = np.asarray(sample) # from list to Numpy array
    wets   = sample[sample>threshold]
    x      = np.sort(wets) # sort ascend by default
    M0hat  = np.mean(x)
    M1hat  = 0.0
    n      = x.size
    for ii in range(n): # N values in the summation
        realii = ii+1
        M1hat = M1hat + x[ii] * (n-realii) 
    M1hat = M1hat/ (n*(n-1))
    csi = 2 - M0hat /(M0hat -2*M1hat)
    beta = 2*M0hat*M1hat/(M0hat -2*M1hat)
    return  csi, beta


def gpd_quant(Fi, csi, beta, mu = 0, ci = False, varcov = []):
    ''' compute GPD quantile q for given non exceedance probabilities in Fi
    GPD with 2 parameters, shape = csi and scale = beta. 
    location can be assigned but default is zero 
    optional: if ci = True also produce the upper and lower confidence 
    intervals obtained under the hyp of normal distribution.
    In this case the covariance matrix of the parameters must be provided
    varcov = variance-covariance matrix of parameters.''' 
    Fi         = np.asarray(Fi)
    is_scalar  = False if Fi.ndim > 0 else True
    Fi.shape   = (1,)*(1-Fi.ndim) + Fi.shape
    if csi != 0.0:
        q = mu + beta/csi *( (1 - Fi)**(-csi) -1) 
    else:
        q = mu - beta * np.log( 1 - Fi)
    q =  q if not is_scalar else  q[0]
    if ci == True:
        # compute std of quantiles using the DELTA METHOD
        # only uncertainty for shape and scale is considered - mu = 0 here.
        m = np.size(Fi)
        qu = np.zeros(m)
        ql = np.zeros(m)
        print('varcov')
        print(varcov)
        for ii in range(m):
            yr = (1-Fi[ii])**(-csi)
            logyr = np.log(1-Fi[ii])
            # der wrt shape and scale
            DEL = np.array([ -beta/csi**2*(yr-1) - beta/csi*yr*logyr,
                            (yr-1)/csi  ])
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

  

                         
def gpd_cdf(q, csi, beta, mu = 0):
    ''' compute GPD non exceedance probabilities Fhi for given quantiles in q
    GPD with 2 parameters, shape = csi and scale = beta. 
    location can be assigned but default is zero ''' 
    q          = np.asarray(q)
    is_scalar  = False if q.ndim > 0 else True
    q.shape    = (1,)*(1-q.ndim) + q.shape 
    if csi != 0.0:
        Fhi = 1 - (1 + csi/beta*(q-mu))**(-1/csi)
    else:
        Fhi = 1 - np.exp( -(q-mu)/beta )
    Fhi        =  Fhi if not is_scalar else  Fhi[0]
    return Fhi 


def gpd_random_quant(length, csi, beta, mu = 0):
    ''' generates quantiles randomly extracted from a 2-par GPD distribution 
    with parameters: shape (csi > 0 if heavy-tailed) and scale (beta);
    location (mu) is zero.
    returns an array of size length or a scalar when length = 1 '''
    Fi = np.random.rand(length)
    if csi != 0.0:
        xi = mu + beta/csi *( (1 - Fi)**(-csi) -1) 
    else:
        xi = mu - beta*np.log( 1 - Fi)
    xi    = xi if length > 1 else xi[0]
    return xi