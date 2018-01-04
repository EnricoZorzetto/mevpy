#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:14:20 2017

@author: enri89
"""


import numpy as np
import matplotlib.pyplot as plt
import mevpy as mev

csi = 0.1
psi = 6
mu = 10
length = 100
sample = mev.gev_random_quant(length, csi, psi, mu)

csih, psih, muh = mev.gev_fit(sample)

# parhat, parstd, varcov = mev.gev_fit_ml(sample, std = True)

parhat, parstd, varcov = mev.gev_fit(sample, how = 'ml', std = True, std_how = 'boot', std_num = 100)



# mat = mev.gev_lmom_bootstrap(sample, ntimes = 100)

## compare confidence intervals for GEV obtained with bootstrap and hessian
#ssizes = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
#ns = np.size(ssizes)
#boot = np.zeros((ns,3))
#hess = np.zeros((ns,3))
#for ii in range(ns):
#    ss = ssizes[ii]
#    sample = mev.gev_random_quant(ss, csi, psi, mu)
#    boot[ii,:] = mev.gev_ml_bootstrap(sample)
#    hess[ii,:] = mev.gev_fit_ml(sample, std = True)[1]
#
#par = 2 # 0 shape 1 scale 2 location    
#plt.plot(ssizes, boot[:,par],'b', label = 'boot')
#plt.plot(ssizes, hess[:,par],'r', label = 'hess')
#plt.legend()
#plt.xlabel('sample size')
#plt.ylabel('par stdv')
#plt.xscale('log')
#plt.show()
    
#Fi = np.linspace(0.2,0.9999,10000)
#q,qu,ql = mev.gev_quant(Fi, csi, psi, mu, ci = True, varcov = varcov)
#Tr = 1/(1-Fi)
#plt.plot(Tr,q,'b', label = 'estimate')
#plt.plot(Tr,qu,'r', label = 'upper')
#plt.plot(Tr,ql,'y', label = 'lower')
#plt.xscale('log')
#plt.xlabel('Return time')
#plt.ylabel('quantile')
#plt.legend()
#plt.show()


## test GPD fit and confidence intervals
#csi = 0.1
#beta = 10.
#lengh = 1000
#sample = mev.gpd_random_quant(length, csi, beta)
#pargat = mev.gpd_fit(sample, how = 'ml', std = True, std_how = 'boot', std_num = 1000)
#print(pargat)
#

## compare confidence intervals for GPD obtained with bootstrap and hessian
#ssizes = np.array([10, 20, 50, 100, 200, 500, 1000])
#ns = np.size(ssizes)
#boot = np.zeros((ns,2))
#hess = np.zeros((ns,2))
#for ii in range(ns):
#    ss = ssizes[ii]
#    sample = mev.gpd_random_quant(ss, csi, beta)
#    parhat1, parstd1, varcov1 = mev.gpd_fit(sample, how = 'pwm', std = True, std_how = 'boot', std_num = 1000)
#    parhat2, parstd2, varcov2 = mev.gpd_fit(sample, how = 'ml', std = True, std_how = 'boot', std_num = 1000)
#    boot[ii, :] = parstd1
#    hess[ii, :] = parstd2
#par = 1 # 0 shape 1 scale 2 location    
#plt.plot(ssizes, boot[:,par],'b', label = 'boot')
#plt.plot(ssizes, hess[:,par],'r', label = 'hess')
#plt.legend()
#plt.xlabel('sample size')
#plt.ylabel('par stdv')
#plt.xscale('log')
#plt.show()
#
#Fi = np.linspace(0.001,0.999,10000)
#csi = 0.1
#beta = 5
#length = 30
#sample = mev.gpd_random_quant(length, csi, beta)
#parhat2, parstd2, varcov2 = mev.gpd_fit(sample, how = 'ml', std = True, std_how = 'hess', std_num = 1000)
#q,qu,ql = mev.gpd_quant(Fi, parhat2[0], parhat2[1], ci = True, varcov = varcov2)
#Tr = 1/(1-Fi)
## Tr = Fi
#plt.plot(Tr,q,'b', label = 'estimate')
#plt.plot(Tr,qu,'r', label = 'upper')
#plt.plot(Tr,ql,'y', label = 'lower')
#plt.xscale('log')
#plt.xlabel('Return time')
#plt.ylabel('quantile')
#plt.legend()
#plt.show()

###################################
## check POT confidence interval
###################################
#csi = 0.1
#beta = 5
#length = 30*366
#sample = mev.gpd_random_quant(length, csi, beta)
#mat = sample.reshape(30,366)
## hess CIs
#boot_ntimes = 100
#parhat, parpot, parstd, varcov = mev.pot_fit(mat, datatype = 'mat', way = 'ea', ea = 3, sp = 0.1, thresh = 10, 
#                              how = 'ml', std = True, std_how = 'hess')
## BOOT CIS
#parhatB, parpotB, parstdB, varcovB = mev.pot_fit(mat, datatype = 'mat', way = 'ea', ea = 3, sp = 0.1, thresh = 10, 
#                              how = 'ml', std = True, std_how = 'boot', std_num = boot_ntimes)
#
#Fi = np.linspace(0.001,0.999,10000)
## Quantiles
#q,qu,ql = mev.pot_quant(Fi, parhat[0], parhat[1], parhat[2], ci = True, parpot = parpot, varcov = varcov)
#qb,qub,qlb = mev.pot_quant(Fi, parhatB[0], parhatB[1], parhatB[2], ci = True, parpot = parpotB, varcov = varcovB)
#
#Tr = 1/(1-Fi)
## Tr = Fi
#plt.plot(Tr,q,'b', label = 'estimate')
#plt.plot(Tr,qu,'r', label = 'upper')
#plt.plot(Tr,ql,'y', label = 'lower')
#plt.plot(Tr,qb,'--b', label = 'estimate boot')
#plt.plot(Tr,qub,'--r', label = 'upper boot')
#plt.plot(Tr,qlb,'--y', label = 'lower boot')
#plt.xscale('log')
#plt.xlabel('Return time')
#plt.ylabel('quantile')
#plt.legend()
#plt.show()


# confidence Intervals for WEI
C = 7
w = 1.0
Fi = np.linspace(0.001,0.9,1000)
length = 1000
xi = mev.wei_random_quant(length,C,w)
# BOOTSTRAP and HESSIAN covariances
parhatb, parstdb, varcovb = mev.wei_fit(sample, how = 'pwm', std = True, std_how = 'boot', std_num = 100)
parhat, parstd, varcov = mev.wei_fit(sample, how = 'pwm', std = True, std_how = 'hess')

# quantiles
qb, qub, qlb = mev.wei_quant(Fi, C, w, ci = True, varcov = varcovb)
q, qu, ql = mev.wei_quant(Fi, C, w, ci = True, varcov = varcov)
Tr = 1/(1-Fi)
plt.plot(Tr,q,'b', label = 'estimate')
plt.plot(Tr,qu,'r', label = 'upper')
plt.plot(Tr,ql,'y', label = 'lower')
plt.plot(Tr,qb,'--b', label = 'estimate bootstrap')
plt.plot(Tr,qub,'--r', label = 'upper bootstrap')
plt.plot(Tr,qlb,'--y', label = 'lower bootstrap')
plt.xscale('log')
plt.xlabel('Return time')
plt.ylabel('quantile')
plt.legend()
plt.show()
