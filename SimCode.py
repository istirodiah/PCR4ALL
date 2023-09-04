#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:13:56 2021

@author: istirodiah
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from scipy.optimize import Bounds
bounds = Bounds(np.zeros((6)), 1*np.ones((6)))


def model(init_vals, period, params, opparams, t):
    S_0, E_0, I_0, A_0, H_0, U_0, L_0, F_0, R_0, D_0, Ns_0 = init_vals
    S, E, I, A, H, U, L, F, R, D, Ns = [S_0], [E_0], [I_0], [A_0], [H_0], [U_0], [L_0], [F_0], [R_0], [D_0], [Ns_0]
    P1, P2, P3, P4, P5, P6, P7, P8 = period
    c, kappa, rho, alpha, delta, sigma, phi, gamma, eta, nu, theta = params
    beta = opparams
    beta = abs(beta)
    
    # print("init ", init_vals)
    
    dt = t[1] - t[0]
    
    for i in t[1:]:
        
        next_Ns  = c.dot(I[-1]*N)*beta*S[-1] + c.dot(A[-1]*N)*beta*S[-1]
        
        next_S = S[-1] - (c.dot(I[-1]*N)*beta*S[-1] + c.dot(A[-1]*N)*beta*S[-1] - P7*R[-1] - P8*F[-1])*dt
        next_E = E[-1] + (c.dot(I[-1]*N)*beta*S[-1] + c.dot(A[-1]*N)*beta*S[-1] - P1*E[-1])*dt
        next_I = I[-1] + (kappa*P1*E[-1] - alpha*P3*I[-1] - (1-alpha)*(1-rho)*P3*I[-1] - (1-alpha)*rho*P3*I[-1])*dt
        next_A = A[-1] + ((1-kappa)*P1*E[-1] - P2*A[-1])*dt
        next_H = H[-1] + (alpha*P3*I[-1] - P4*H[-1] + (1-gamma)*eta*P6*L[-1])*dt
        next_U = U[-1] + (delta*P4*H[-1] - P5*U[-1])*dt
        next_L = L[-1] + ((1-alpha)*rho*P3*I[-1] + (1-delta)*phi*P4*H[-1] + (1-theta)*sigma*P5*U[-1] - P6*L[-1])*dt
        next_F = F[-1] + ((1-nu)*P2*A[-1] + (1-alpha)*(1-rho)*P3*I[-1] + (1-delta)*(1-phi)*P4*H[-1] + (1-theta)*(1-sigma)*P5*U[-1] - P8*F[-1])*dt
        next_R = R[-1] + ((1-gamma)*(1-eta)*P6*L[-1] - P7*R[-1])*dt
        next_D = D[-1] + (gamma*P6*L[-1] + theta*P5*U[-1] + nu*P2*A[-1])*dt
        
        S  = np.vstack((S, next_S))
        E  = np.vstack((E, next_E))
        I  = np.vstack((I, next_I))
        A  = np.vstack((A, next_A))
        H  = np.vstack((H, next_H))
        U  = np.vstack((U, next_U))
        L  = np.vstack((L, next_L))
        F  = np.vstack((F, next_F))
        R  = np.vstack((R, next_R))
        D  = np.vstack((D, next_D))
        Ns = np.vstack((Ns, next_Ns))
    
    return S, E, I, A, H, U, L, F, R, D, Ns


def cost(opparams, modelparams, data, i, iw):
    init_vals, period, params, N, t = modelparams
    S, E, I, A, H, U, L, F, R, D, Ns = model(init_vals, period, params, opparams, t)
    dum = 0
    for j in range(iw):
        sim = Ns[(iw - j)*7,:] * N
        q = (sim - data[i-j])**2 * 1./max(data[i-j])
        dum = dum + sum(q)
    return dum
    


file = pd.ExcelFile('Data.xlsx')
df1 = file.parse(5)#(6)# #Infection
df2 = file.parse(4) #CumDeath
df3 = file.parse(2) #Death

idata  = df1.values[:, 1:]
cddata = df2.values[:, 1:]
ddata  = df3.values[:, 1:]


it = 68# 60
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters
P1 = 1./3 #5
P2 = 1./7 #9
P3 = 1./4
P4 = 1./7
P5 = 1./10
P6 = 1./14
P7 = 1./90
P8 = 1./360


c   = np.array([[5.090928e-07, 1.016800e-07, 9.904523e-08, 6.037230e-08, 3.330160e-08, 1.418771e-08],
                [1.016800e-07, 4.277460e-07, 7.586532e-08, 7.709942e-08, 3.231887e-08, 1.384713e-08],
                [9.904523e-08, 7.586532e-08, 2.650871e-07, 1.042291e-07, 3.812121e-08, 2.656602e-08],
                [6.037230e-08, 7.709942e-08, 1.042291e-07, 1.308419e-07, 5.894746e-08, 3.369221e-08],
                [3.330160e-08, 3.231887e-08, 3.812121e-08, 5.894746e-08, 1.187878e-07, 6.106433e-08],
                [1.418771e-08, 1.384713e-08, 2.656602e-08, 3.369221e-08, 6.106433e-08, 8.840770e-08]])

kappa = np.array([0.4, 0.4, 0.8, 0.8, 0.8, 0.8])
rho   = np.array([0.0001, 0.001, 0.1, 0.3, 0.4, 0.5])
alpha = np.array([0.0001, 0.012, 0.032, 0.049, 0.152, 0.273])
delta = np.array([0.05, 0.05, 0.063, 0.122, 0.303, 0.709])
sigma = 0.10
phi   = 0.20
eta   = 0.40
theta = np.array([9.16091458e-06, 9.16091458e-06, 3.95604731e-04, 6.58625427e-05, 4.96556745e-03, 2.36730805e-02])
gamma = np.array([0.0, 0.0, 0.0, 3.13666771e-04, 0.0, 1.60165839e-04])
nu    = np.array([0.0, 0.0, 0.00, 2.13666771e-04, 1.87118193e-02, 3.36108712e-02])
period = P1, P2, P3, P4, P5, P6, P7, P8
params = c, kappa, rho, alpha, delta, sigma, phi, gamma, eta, nu, theta

# file1 = pd.ExcelFile('Dum.xlsx')
# df5 = file1.parse(11)#(11)# #rate
# beta = df5.values[:, :]
# beta = beta.astype(float)

beta = 0.001*np.ones((6))

### Initial States KW 5
N = 83166711         # Bundesamt data
# e = 660             #kw9
# i = 139             #kw9
# r = 15              #kw9
e = 55             #kw5
i = 4              #kw5
r = 0              #kw5
a = i
h = 0.*i           
u = 0
l = 0            
f = 0
d = 0


N = 83166711  * np.array([0.0470, 0.0920, 0.2280, 0.3500, 0.2150, 0.0680])
i = i * np.array([0., 0., 0.25, 0.25, 0.25, 0.25 ])#kw5

E_0 = e*1./N * np.array([0.0909, 0., 0.5454, 0.3636, 0., 0.]) # KW5
I_0 = i*1./N * np.array([0.4, 0.4, 0.8, 0.8, 0.8, 0.8])
A_0 = i*1./N * np.array([0.6, 0.6, 0.2, 0.2, 0.2, 0.2])
H_0 = h*1./N * np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.3])
U_0 = u*1./N * np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.3])
L_0 = l*1./N * np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.3])
F_0 = f*1./N * np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
R_0 = r*1./N * np.array([0.1176, 0.0, 0.3529, 0.5294, 0.0, 0.0])
D_0 = d*1./N * np.array([0.0001, 0.0000, 0.0026, 0.0487, 0.3236, 0.6250])
S_0 = N*1./N - (E_0+I_0+A_0+H_0+U_0+L_0+F_0+R_0+D_0)
Ns_0 = E_0

init_vals   = S_0, E_0, I_0, A_0, H_0, U_0, L_0, F_0, R_0, D_0, Ns_0
modelparams = init_vals, period, params, N, t


b = np.zeros((len(times), 6))
S, E, I, A, H, U, L, F, R, D, Ns = [S_0], [E_0], [I_0], [A_0], [H_0], [U_0], [L_0], [F_0], [R_0], [D_0], [Ns_0]

# b = rate

iw = 1
for i, _ in enumerate(times):
    if i > 0 and i%iw==0:
        dummyt = np.linspace(times[i-iw], times[i], int((times[i]-times[i-iw])/dt)+1)
        modelparams = init_vals, period, params, N, dummyt
        optimizer = opt.minimize(cost, beta, args=(modelparams, idata, i, iw), tol=1e-10)
        # optimizer = opt.minimize(cost, beta[i], args=(modelparams, idata, i, iw), tol=1e-10)
        b[i] = optimizer.x 
        
        if np.any(b[i]<0) == True:
            optimizer = opt.minimize(cost, b[i], args=(modelparams, idata, i, iw), tol=1e-10, bounds=bounds)
            
        for j in range(iw):
            b[i-j] = optimizer.x
        
        
        nS, nE, nI, nA, nH, nU, nL, nF, nR, nD, nNs = model(init_vals, period, params, b[i], dummyt)
        init_vals = nS[-1], nE[-1], nI[-1], nA[-1], nH[-1], nU[-1], nL[-1], nF[-1], nR[-1], nD[-1], nNs[-1]
        
        S = np.vstack((S, nS[1:,:]))
        E = np.vstack((E, nE[1:,:]))
        I = np.vstack((I, nI[1:,:]))
        A = np.vstack((A, nA[1:,:]))
        H = np.vstack((H, nH[1:,:]))
        U = np.vstack((U, nU[1:,:]))
        L = np.vstack((L, nL[1:,:]))
        F = np.vstack((F, nF[1:,:]))
        R = np.vstack((R, nR[1:,:]))
        D = np.vstack((D, nD[1:,:]))
        Ns = np.vstack((Ns, nNs[1:,:]))
        
        ind = i
