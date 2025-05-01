r"""
NAME
    main_ai_model.py - Computes a steady state equilibrium for an artificial economy that produces and uses AI

SYNOPSIS
    Execute at the Python prompt as >>> import main_ai_model,
    or run as `python3 main_ai_model.py` at the terminal

AUTHOR
    Gonzalo F. de Cordoba & Alvaro Mezquita

LICENSE
    Copyright © 2024 License GPLv3+: GNU GPL version 3 or later
    <http://gnu.org/licenses/gpl.html>.
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.

DESCRIPTION
    This program solves the steady state of the following economy:

    A representative consumer maximizes discounted utility with a set of
    constraints over output and the amount of AI that can get out of the
    existing AI and a sacrifice of capital and labor augmenting AI.

    Max beta^{t-1} log C{t}
    s.t.   C_{t} + K_{t+1} - (1-\delta_{k})K_{t} = FK_{f,t}^{1-\alpha}
    \left[(1-\sigma)L_{f,t}^{-\gamma} + \sigma A_{t}^{-\gamma}
    \right]^{-\frac{\alpha}{\gamma}} = A_{t+1}
    DL_{at}^{\phi}K_{at}^{\theta} + (1-\delta_{A})A_{t} 
    K_{at} + K_{ft} = K_{t} 
    L_{at} + L_{ft} = \bar{L}
    A_{0}, K_{0} given

    A profit maximizing firm:
    Max Y_{t}(K_{f,t}, L_{f, t}, A_{t}) - w_{t}L_{ft} - r_{t}^{k}K_{ft}
    - r_{t}^{A}A_{t}

    Operates a technology:
    Y_{t} = FK_{f,t}^{1-\alpha} \left[(1-\sigma)L_{f,t}^{-\gamma} +
            \sigma A_{t}^{-\gamma} \right]^{-\frac{\alpha}{\gamma}}
"""


# Importing libraries 
import numpy as np 
from scipy.optimize import root  

# Parameters of the model
beta = 0.99 
delta_a = 0.05 
delta_k = 0.01 
D = 1 
phi = 0.75 
theta = 1 - phi 
F = 1 
alpha = 0.80 
gamma = 0.5
sigma = 0.5 
Lbar = 1 

parameters = np.array([beta,delta_a,delta_k,D,phi,theta,F,alpha,gamma,sigma,Lbar])

# Secant method seed 
Ka_ss_0 = 10 
La_ss_0 = 0.5 
Kf_ss_0 = 10

ss_0 = np.array([Ka_ss_0, La_ss_0, Kf_ss_0])  

# Defining equation system of Steady State
def ss(seed, param): 
    # Unpacking parameters
    beta,delta_a,delta_k,D,phi,theta,F,alpha,gamma,sigma,Lbar = param 

    # Unpacking variables
    Ka, La, Kf = seed 

    # Equations cleaning
    Lf = Lbar-La 
    A = D * La**phi * Ka**theta /delta_a 
    Q = (1-sigma) * Lf**(-gamma) + sigma * A**(-gamma) 
    Rk = F * (1-alpha) * Kf**(-alpha) * Q**(-alpha/gamma) 
    W = F * alpha * (1-sigma) * Kf**(1-alpha) * Q**(-alpha/gamma-1) * Lf**(-gamma-1) 
    Ra = F * alpha * sigma * Kf**(1-alpha) * Q**(-alpha/gamma-1) * A**(-gamma-1) 
    K = Ka + Kf 
    C =  F * Kf**(1-alpha) * ((1-sigma) * Lf**(-gamma) + sigma * A**(-gamma))**(-alpha/gamma) - delta_k*K
    lambda_ss = 1/C 
    kappa = lambda_ss/beta - lambda_ss*(1-delta_k) 
    mu = kappa/(theta * D * La**phi * Ka**(theta-1)) 
    psi = mu * phi * D * Ka**theta * La**(phi-1) 

    # Remaining equations 
    f1 = lambda_ss * (1 - alpha) * F * Kf**(-alpha) * Q**(-alpha/gamma) - kappa
    f2 = lambda_ss * alpha * (1 - sigma) * F * Kf**(1 - alpha) * Q**(-alpha/gamma - 1) * Lf**(-gamma - 1) - psi
    f3 = alpha * sigma * F * Kf**(1 - alpha) * Q**(-alpha/gamma - 1) * A**(-gamma - 1) - mu/beta + mu * (1 - delta_a)

    return np.array([f1,f2,f3]) 


# Computing the solution for Ka, La and Kf
solution = root(ss,ss_0,args=(parameters,)) 
Ka_sol, La_sol, Kf_sol = solution.x 

# Computing the rest of the variables 
Lf = Lbar-La_sol 
A = D * La_sol**phi * Ka_sol**theta /delta_a 
Q = (1-sigma) * Lf**(-gamma) + sigma * A**(-gamma) 
Rk = F * (1-alpha) * Kf_sol**(-alpha) * Q**(-alpha/gamma) 
W = F * alpha * (1-sigma) * Kf_sol**(1-alpha) * Q**(-alpha/gamma-1) * Lf**(-gamma-1) 
Ra = F * alpha * sigma * Kf_sol**(1-alpha) * Q**(-alpha/gamma-1) * A**(-gamma-1) 
K = Ka_sol + Kf_sol 
C =  F * Kf_sol**(1-alpha) * ((1-sigma) * Lf**(-gamma) + sigma * A**(-gamma))**(-alpha/gamma) - delta_k*K
lambda_ss = 1/C 
kappa = lambda_ss/beta - lambda_ss*(1-delta_k) 
mu = kappa/(theta * D * La_sol**phi * Ka_sol**(theta-1)) 
psi = mu * phi * D * Ka_sol**theta * La_sol**(phi-1)  

# Printing results 
print("Capital destined to AI industry: ",Ka_sol) 
print("Capital destined to Final Good industry: ",Kf_sol) 
print("Labor destined to Final Good industry: ", Lf) 
print("Labor destined to AI industry: ",La_sol) 
print("Level of AI: ",A) 
print("Capital returns: ",Rk) 
print("AI returns: ", Ra) 
print("Wages level: ",W) 
print("Level of Capital: ",K) 
print("Level of Consumption: ",C) 
