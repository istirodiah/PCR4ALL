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
bounds = Bounds(np.zeros((18)), 1*np.ones((18)))

def model(opparams, init_vals, period, params, t, index):
    S_0, V_0, B_0, E_0, Ev_0, Eb_0, I_0, A_0, Iv_0, Ib_0, H_0, Hv_0, Hb_0, U_0, Uv_0, Ub_0, L_0, F_0, R_0, D_0, Ns_0, Nh_0, Nu_0 = init_vals
    S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu = [S_0], [V_0], [B_0], [E_0], [Ev_0], [Eb_0], [I_0], [A_0], [Iv_0], [Ib_0], [H_0], [Hv_0], [Hb_0], [U_0], [Uv_0], [Ub_0], [L_0], [F_0], [R_0], [D_0], [Ns_0], [Nh_0], [Nu_0]
    P1, P2, P3, P4, P5, P6, P7, P8, P9 = period
    c, kappa, rho, delta, sigma, phi, gamma, eta, nu, delta, epsilon, epsilonv = params
    
    beta  = opparams[:6]    
    theta = opparams[6:12]  
    alpha = opparams[12:18] 

    betav  = beta * np.array([0.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    betab  = beta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25]) # normal case
    
    thetav = theta * np.array([0.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    thetab = theta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25]) # normal case
    
    alphav = alpha * np.array([0.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    alphab = alpha * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25]) # normal case
    
    deltav = delta * np.array([0.0, 0.2 , 0.5 , 0.5, 0.6, 0.6])
    deltab = delta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25]) # normal case
  
    dt = (t[1] - t[0])*1./7
    
    for i in t[1:]:
        
        next_Ns = Ns[-1] + (c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*beta*S[-1] +  c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*betav*V[-1] + c.dot((I[-1]+A[-1]+Iv[-1]+Ib[-1])*N)*betab*B[-1])*dt
        next_Nu = Nu[-1] + (delta*P4*H[-1] + deltav*P4*Hv[-1] + deltab*P4*Hb[-1])*dt
        
        
        next_S  = S[-1] - (c.dot(I[-1]*N)*beta*S[-1] + c.dot(A[-1]*N)*beta*S[-1] + c.dot(Iv[-1]*N)*beta*S[-1] + c.dot(Ib[-1]*N)*beta*S[-1] + epsilon*P9*S[-1] - P7*R[-1] - P8*F[-1])*dt
        next_V  = V[-1] - (c.dot(I[-1]*N)*betav*V[-1] + c.dot(A[-1]*N)*betav*V[-1] + c.dot(Iv[-1]*N)*betav*V[-1] + c.dot(Ib[-1]*N)*betav*V[-1] - epsilon*P9*S[-1] + epsilonv*P9*V[-1])*dt
        next_B  = B[-1] - (c.dot(I[-1]*N)*betab*B[-1] + c.dot(A[-1]*N)*betab*B[-1] + c.dot(Iv[-1]*N)*betab*B[-1] + c.dot(Ib[-1]*N)*betab*B[-1] - epsilonv*P9*V[-1])*dt
        next_E  = E[-1] + (c.dot(I[-1]*N)*beta*S[-1] + c.dot(A[-1]*N)*beta*S[-1] + c.dot(Iv[-1]*N)*beta*S[-1] + c.dot(Ib[-1]*N)*beta*S[-1] - P1*E[-1])*dt
        next_Ev = Ev[-1] + (c.dot(I[-1]*N)*betav*V[-1] + c.dot(A[-1]*N)*betav*V[-1] + c.dot(Iv[-1]*N)*betav*V[-1] + c.dot(Ib[-1]*N)*betav*V[-1] - P1*Ev[-1])*dt
        next_Eb = Eb[-1] + (c.dot(I[-1]*N)*betav*V[-1] + c.dot(A[-1]*N)*betav*V[-1] + c.dot(Iv[-1]*N)*betav*V[-1] + c.dot(Ib[-1]*N)*betab*B[-1] - P1*Eb[-1])*dt
        next_I  = I[-1] + (kappa*P1*E[-1] - alpha*P3*I[-1] - (1-alpha)*(1-rho)*P3*I[-1] - (1-alpha)*rho*P3*I[-1])*dt
        next_A  = A[-1] + ((1-kappa)*P1*E[-1] - P2*A[-1])*dt
        next_Iv = Iv[-1] + (P1*Ev[-1] - P3*Iv[-1])*dt
        next_Ib = Ib[-1] + (P1*Eb[-1] - P3*Ib[-1])*dt
        next_H  = H[-1] + (alpha*P3*I[-1] - P4*H[-1] + (1-gamma)*eta*P6*L[-1])*dt
        next_Hv = Hv[-1] + (alphav*P3*Iv[-1] - P4*Hv[-1])*dt
        next_Hb = Hb[-1] + (alphab*P3*Ib[-1] - P4*Hb[-1])*dt
        next_U  = U[-1] + (delta*P4*H[-1] - P5*U[-1])*dt
        next_Uv = Uv[-1] + (deltav*P4*Hv[-1] - P5*Uv[-1])*dt
        next_Ub = Ub[-1] + (deltab*P4*Hb[-1] - P5*Ub[-1])*dt
        next_L  = L[-1] + ((1-alpha)*rho*P3*I[-1] + (1-delta)*phi*P4*H[-1] + (1-theta)*sigma*P5*U[-1] - P6*L[-1])*dt
        next_F  = F[-1] + ((1-nu)*P2*A[-1] + (1-alpha)*(1-rho)*P3*I[-1] + (1-delta)*(1-phi)*P4*H[-1] + (1-theta)*(1-sigma)*P5*U[-1] - P8*F[-1] + (1-alphav)*P3*Iv[-1] + (1-deltav)*P4*Hv[-1] +(1-thetav)*P5*Uv[-1] + (1-alphab)*P3*Ib[-1] + (1-deltab)*P4*Hb[-1] +(1-thetab)*P5*Ub[-1])*dt
        next_R  = R[-1] + ((1-gamma)*(1-eta)*P6*L[-1] - P7*R[-1])*dt
        next_D  = D[-1] + (gamma*P6*L[-1] + theta*P5*U[-1] + nu*P2*A[-1] + thetav*P5*Uv[-1] + thetab*P5*Ub[-1])*dt
        
        next_Nh = next_H + next_Hv + next_Hb
        
        Ns = np.vstack((Ns, next_Ns))
        Nh = np.vstack((Nh, next_Nh))
        Nu = np.vstack((Nu, next_Nu))
        
        S  = np.vstack((S, next_S))
        V  = np.vstack((V, next_V))
        B  = np.vstack((B, next_B))
        E  = np.vstack((E, next_E))
        Ev = np.vstack((Ev, next_Ev))
        Eb = np.vstack((Eb, next_Eb))
        I  = np.vstack((I, next_I))
        A  = np.vstack((A, next_A))
        Iv = np.vstack((Iv, next_Iv))
        Ib = np.vstack((Ib, next_Ib))
        H  = np.vstack((H, next_H))
        Hv = np.vstack((Hv, next_Hv))
        Hb = np.vstack((Hb, next_Hb))
        U  = np.vstack((U, next_U))
        Uv = np.vstack((Uv, next_Uv))
        Ub = np.vstack((Ub, next_Ub))
        L  = np.vstack((L, next_L))
        F  = np.vstack((F, next_F))
        R  = np.vstack((R, next_R))
        D  = np.vstack((D, next_D))
    
    return S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu




file = pd.ExcelFile('Forecast.xlsx')
df1  = file.parse(0) #Infection
df2  = file.parse(1) #CumDeath
df3  = file.parse(2) #Death
df4  = file.parse(3) #Hospitalization
df5  = file.parse(4) #ICU

idata  = df1.values[144:149, 1:7]
cddata = df2.values[144:149, 1:7]
ddata  = df3.values[144:149, 1:7]
hdata  = df4.values[144:149, 1:7]
udata  = df5.values[144:149, 1:7]
   
it = 52
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters
P1 = 1./3
P2 = 1./7
P3 = 1./4 
P4 = 1./7 
P5 = 1./5
P6 = 1./14
P7 = 1./90
P8 = 1./180
P9 = 1./14

c   = np.array([[2.021E-07, 7.763E-08, 4.520E-08, 4.789E-08, 2.909E-08, 4.994E-08],
                [7.763E-08, 5.721E-08, 2.569E-08, 2.176E-08, 1.171E-08, 2.647E-08],
                [4.520E-08, 2.569E-08, 2.942E-08, 1.908E-08, 1.274E-08, 4.122E-08],
                [4.789E-08, 2.176E-08, 1.908E-08, 1.418E-08, 8.350E-09, 1.447E-08],
                [2.909E-08, 1.171E-08, 1.274E-08, 8.350E-09, 1.488E-08, 1.441E-08],
                [4.994E-08, 2.647E-08, 4.122E-08, 1.447E-08, 1.441E-08, 2.858E-08]])


kappa    = np.array([0.4, 0.4, 0.8, 0.8, 0.8, 0.8])
rho      = np.array([0.0001, 0.001, 0.1, 0.3, 0.4, 0.5])
alpha    = np.array([0.025, 0.003, 0.012, 0.025, 0.099, 0.262])
delta    = np.array([0.01, 0.05, 0.063, 0.376, 0.545, 0.709]) 
sigma    = 0.10
phi      = 0.20
eta      = 0. #0.40
beta     = np.array([0.0474986, 0.09716402, 0.07156201, 0.0664224, 0.04255577, 0.0645]) 
theta    = np.array([9.16091458e-06, 9.16091458e-06, 3.95604731e-04, 6.58625427e-05, 4.96556745e-03, 2.36730805e-02])
gamma    = np.array([0.0, 0.0, 0.0, 3.13666771e-04, 0.0, 1.60165839e-04])
nu       = np.array([0.0, 0.0, 0.00, 2.13666771e-05, 1.87118193e-04, 8.36108712e-03])
epsilon  = np.array([0., 0., 0., 0., 0., 0.]) #np.array([0., 0.01, 0.01, 0.01, 0.01, 0.01])
betav    = beta * np.array([0.0, 0.01 , 0.5 , 0.5, 0.6, 0.6])
thetav   = theta * np.array([0.0, 0.0 , 0.5 , 0.5, 0.4, 0.4]) #np.array([0, 0, 3.9e-05, 6.5e-06, 4.9e-03, 2.3e-02])
deltav   = delta * np.array([0.0, 0.0 , 0.5 , 0.5, 0.4, 0.4]) #np.array([0, 0, 0.01, 0.01, 0.1, 0.1])
alphav   = alpha * np.array([0., 0.001, 0.4, 0.4, 0.3, 0.2])
epsilonv = np.array([0., 0., 0., 0., 0., 0.]) #np.array([0., 0., 0., 0.001, 0.005, 0.005])
betab    = beta * np.array([0.0, 0.0 , 0.4, 0.4, 0.25, 0.25])
thetab   = theta * np.array([0.0, 0.0 , 0.4 , 0.4, 0.1, 0.1])
deltab   = delta * np.array([0.0, 0.0 , 0.4 , 0.4, 0.1, 0.1])
alphab   = alpha * np.array([0.0, 0.0 , 0.25 , 0.2, 0.2, 0.1])

period  = P1, P2, P3, P4, P5, P6, P7, P8, P9

### Initial States KW 14
N = 83166711         # worldometer data
N = N * np.array([0.0470, 0.0920, 0.2280, 0.3500, 0.2150, 0.0680])

file1 = pd.ExcelFile('ScenarioNew.xlsx')
df4 = file1.parse(2) #Init
init_vals = df4.values[:,1:7]
init_vals = init_vals.astype(float)
init_vals[10] = 0.8*hdata[-1]/N
init_vals[11] = 0.1*hdata[-1]/N
init_vals[12] = 0.1*hdata[-1]/N
init_vals[21] = hdata[-1]/N

df5 = file1.parse(3)#12 #Coef
coef = df5.values[6:,1:]
coef = coef.astype(float)

df6 = file1.parse(1)#10 #Rate
par = df6.values[-1,1:]
par = par.astype(float)

epsilon  = np.array([0., 0.02, 0.05, 0.05, 0.05, 0.05])
epsilonv = np.array([0., 0., 0., 0., 0., 0.0])
# epsilonv = np.array([0., 0., 0.04, 0.04, 0.05, 0.05])

params   = c, kappa, rho, delta, sigma, phi, gamma, eta, nu, delta, epsilon, epsilonv
modelparams = init_vals, period, params, N, t
S_0, V_0, B_0, E_0, Ev_0, Eb_0, I_0, A_0, Iv_0, Ib_0, H_0, Hv_0, Hb_0, U_0, Uv_0, Ub_0, L_0, F_0, R_0, D_0, Ns_0, Nh_0, Nu_0 = init_vals
S, V, B, E, Ev, Eb, I, A, Iv, Ib, H, Hv, Hb, U, Uv, Ub, L, F, R, D, Ns, Nh, Nu = [S_0], [V_0], [B_0], [E_0], [Ev_0], [Eb_0], [I_0], [A_0], [Iv_0], [Ib_0], [H_0], [Hv_0], [Hb_0], [U_0], [Uv_0], [Ub_0], [L_0], [F_0], [R_0], [D_0], [Ns_0], [Nh_0], [Nu_0]
 
ft  = np.arange(0, 52*7, 7)

w1 = -1*np.log(0.4)/(6*30)# -1*np.log(0.7)/(6*30)
w2 = 0#-1*np.log(0.8)/(6*30)# 1

                 
fS, fV, fB, fE, fEv, fEb, fI, fA, fIv, fIb, fH, fHv, fHb, fU, fUv, fUb, fL, fF, fR, fD, fNs, fNh, fNu = init_vals
for i, k in enumerate(ft):
    dummyt = np.linspace(0, 7, 8)
    
    par[:6]    = par[:6]*coef[i,:6]*(1+w1)  
    par[6:12]  = par[6:12]*coef[i,6:12]*(1+w2)
    par[12:18] = par[12:18]*coef[i,12:18]*(1+w2)
    par[par>1] = 1
    
    
    nS, nV, nB, nE, nEv, nEb, nI, nA, nIv, nIb, nH, nHv, nHb, nU, nUv, nUb, nL, nF, nR, nD, nNs, nNh, nNu = model(par, init_vals, period, params, dummyt, i)
    init_vals = nS[-1], nV[-1], nB[-1], nE[-1], nEv[-1], nEb[-1], nI[-1], nA[-1], nIv[-1], nIb[-1], nH[-1], nHv[-1], nHb[-1], nU[-1], nUv[-1], nUb[-1], nL[-1], nF[-1], nR[-1], nD[-1], nNs[-1], nNh[-1], nNu[-1]

    fS  = np.vstack((fS, nS[1:,:]))
    fV  = np.vstack((fV, nV[1:,:]))
    fB  = np.vstack((fB, nB[1:,:]))
    fE  = np.vstack((fE, nE[1:,:]))
    fEv = np.vstack((fEv, nEv[1:,:]))
    fEb = np.vstack((fEb, nEb[1:,:]))
    fI  = np.vstack((fI, nI[1:,:]))
    fA  = np.vstack((fA, nA[1:,:]))
    fIv = np.vstack((fIv, nIv[1:,:]))
    fIb = np.vstack((fIb, nIb[1:,:]))
    fH  = np.vstack((fH, nH[1:,:]))
    fHv = np.vstack((fHv, nHv[1:,:]))
    fHb = np.vstack((fHb, nHb[1:,:]))
    fU  = np.vstack((fU, nU[1:,:]))
    fUv = np.vstack((fUv, nUv[1:,:]))
    fUb = np.vstack((fUb, nUb[1:,:]))
    fL  = np.vstack((fL, nL[1:,:]))
    fF  = np.vstack((fF, nF[1:,:]))
    fR  = np.vstack((fR, nR[1:,:]))
    fD  = np.vstack((fD, nD[1:,:]))
    fNs = np.vstack((fNs, nNs[1:,:]))
    fNh = np.vstack((fNh, nNh[1:,:]))
    fNu = np.vstack((fNu, nNu[1:,:]))

zN  = fNs[::7]*N
zNh = fNh[::7]*N
zD  = fD[::7]*N
zI  = fI[::7]*N
zNs = np.zeros((53,6))

for i in range(52):
    # zNd[i+1] = zD[i+1] - zD[i]
    zNs[i+1] = zN[i+1]-zN[i]
zNs[0] = idata[-1]
# zNd[0] = ddata[-1]


color = ['orange', 'green', 'cyan', 'blue', 'grey', 'red']
label = ['0-4 years', '5-14 years', '15-34 years', '35-59 years', '60-79 years', '80+ years']

for i in range(6):
    plt.plot(times/7,zNs[:,i], color=color[i], label=label[i])
plt.xlabel('Week')
plt.ylabel('New Cases')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
plt.show()

