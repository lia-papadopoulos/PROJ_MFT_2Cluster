
import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt


'''
set of all functions needed for MFT calculation of LIF model
assumes m dynamical populations 
can work with exponential synapses
assumes all neurons in the same population receive the same number of inputs
no quenched heterogeneity

parameters are:
    Vr, Vth:                reset and threshold voltages, vectors of length m
    tau_r, tau_m, tau_s:    refractory, membrane, synaptic time constants; vectors of length m
    nu_ext:                 external input rates; vector of length m
    ext_variance:           0 or 1, indicating that external inputs should be considered deterministic or poisson; vector of length m
    Jab, Jab_ext:           recurrent and external synaptic weight matrices; size mxm and mxm, respectively
    Cab, Cab_ext:           recurrent and external degree matrices; size mxm and mxm, respectively
    nu_vec:                 initial guess for the mft solution; vector of size m
'''

     
#%% FIRING RATE INTEGRAND (INCLUDES SQRT PI FACTOR)

def fcn_firingRate_integrand(u):

    '''
    define integrand for LIF firing rate
    this definition already includes the sqrt pi factor

    Note: this way causes problems b/c of multiplication of large # by small #
    LIF_integrand = np.sqrt(np.pi)*np.exp(u**2)*(1+special.erf(u))
    '''
        
    # use asymptotic expansion of erfc if u is very negative
    if u < -25:
        print('using asymptotic expansion of erfc')
        LIF_integrand = (1-1/(2*u**2)+3/(4*u**4)-15/(8*u**6))*(-1/u)
    # otherwise, use the exact form (but write in a way that helps avoid numerical issues)
    else:
        A = u*u
        B = special.erfc(-u)
        LIF_integrand = np.sqrt(np.pi)*np.exp(A + np.log(B))
    
    return LIF_integrand
    

#%% BRUNEL-SERGI CORRECTION FOR FIRING RATE INTEGRAL

def fcn_BrunelSergi_correction(tau_m, tau_s):

    '''
    Brunel-Sergi correction for firing rate integral
    Takes into account effects of synaptic time constat
    '''

    a = -special.zeta(1/2)/np.sqrt(2) 
    BS = a*np.sqrt(tau_s/tau_m)
    return BS

#%% UPPER TRANSFER FUNCTION LIMIT

def fcn_thresh_lim(Mu, sigma, Vth):
        
    '''
    define upper limit of LIF transfer function
    '''
        
    thresh_lim = ((Vth - Mu) / sigma)
    return thresh_lim


#%% LOWER TRANSFER FUNCTION LIMIT

def fcn_reset_lim(Mu, sigma, Vr):
        
    '''
    define upper limit of LIF transfer function
    '''

    reset_lim = ((Vr - Mu) / sigma)
    return reset_lim
    

#%% COMPUTE FIRING RATE

def fcn_compute_rate(Vr, Vth, Mu, sigma, tau_r, tau_m, tau_s):
        
    '''
    compute LIF firing rate
    '''
    
    # upper and low integration limits
    BS = fcn_BrunelSergi_correction(tau_m, tau_s)
    lower_lim = fcn_reset_lim(Mu, sigma, Vr) + BS  
    upper_lim = fcn_thresh_lim(Mu, sigma, Vth) + BS
    
    # compute the integral
    integral, _ = \
    integrate.quad(fcn_firingRate_integrand, lower_lim, upper_lim, \
                    epsabs=1e-12, epsrel=1e-12)
        
    inv_rate = (tau_r + tau_m*integral)
    nu = 1/inv_rate 
    return nu
    
#%% COMPUTE INVERSE OF FIRING RATE

def fcn_compute_inv_rate(Vr, Vth, Mu, sigma, tau_r, tau_m, tau_s):
    
    '''
    compute inverse LIF firing rate
    '''

    # upper and lower integrate limits
    BS = fcn_BrunelSergi_correction(tau_m, tau_s)
    lower_lim = fcn_reset_lim(Mu, sigma, Vr) + BS  
    upper_lim = fcn_thresh_lim(Mu, sigma, Vth) + BS
    
    # compute the integral
    integral, _ = \
    integrate.quad(fcn_firingRate_integrand, lower_lim, upper_lim, \
                    epsabs=1e-12, epsrel=1e-12)
        
    inv_rate = (tau_r + tau_m*integral)
    return inv_rate
    

#%% COMPUTE MEAN INPUT 

def fcn_compute_Mu(nu, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m):   
    
    '''
    compute mean input
    '''
           
    # MEAN OF INPUT TO EACH POPULATION
    mu_recurrent_vec = np.matmul( (Cab*Jab) , nu) * tau_m
    mu_external_vec = Jab_ext*Cab_ext*tau_m*nu_ext
    mu_vec =  mu_recurrent_vec + mu_external_vec
                          
    return mu_vec           
    

#%% COMPUTE VARIANCE INPUT
 
def fcn_compute_Sigma2(nu, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m, ext_variance):   
    
    '''
    compute variance of the input
    '''

    # VARIANCE OF INPUT TO EACH POPULATION
    sig2_recurrent_vec = np.matmul( (Cab*Jab*Jab) , nu) * tau_m
    sig2_external_vec = Jab_ext*Jab_ext*Cab_ext*tau_m*nu_ext*ext_variance

    sig2_vec =  sig2_recurrent_vec + sig2_external_vec
    
    return sig2_vec


#%% COMPUTE STATIONARY RATES BY SOLVING DYNAMICAL EQUATIONS

def fcn_compute_MFT_rates_DynEqs(nSteps, dt, T, stop_thresh, plot, \
                                 tau_r, tau_m, tau_s, Vr,  Vth,  \
                                 Jab, Cab, Jab_ext, Cab_ext, ext_variance, nu_ext, nu_vec_in):
    
    '''
    all parameters (e.g. time constats, threshold voltages, etc) should be vectors
    of length = # dynamical populations
    '''
    
    ##### SETUP
    
    # initialize
    n_pops = np.size(nu_vec_in)
    
    nu = np.zeros((n_pops, nSteps+1))
    
    
    #### MAIN LOOP

    # set initial conditions
    nu[:,0] = nu_vec_in.copy()
    
    # time loop
    for i in range(0,nSteps,1):
        
        # compute information for next time step
        
        # compute mean of inputs
        Mu = fcn_compute_Mu(nu[:,i], nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m)
    
        # compute variance of inputs
        Sigma2 = fcn_compute_Sigma2(nu[:,i], nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m, ext_variance)
            
        # compute standard deviations
        sigma = np.sqrt(Sigma2)
        
        # compute output rates
        phi = np.zeros(n_pops)
        
        for pop_ind in range(0,n_pops):
            
            phi[pop_ind] = fcn_compute_rate(Vr[pop_ind], Vth[pop_ind], \
                                            Mu[pop_ind], sigma[pop_ind], \
                                            tau_r[pop_ind], tau_m[pop_ind], tau_s[pop_ind])
            

        # update rates
        nu[:,i+1] = nu[:,i] + (-nu[:,i]/T + phi/T)*dt
        
        # check tolerances
        nu_check = all(abs(nu[:,i+1]-nu[:,i]) < stop_thresh)
        
        if (nu_check == True):
            
            # delete remaining elements
            nu = np.delete(nu, np.arange(i+1,nSteps+1), 1)
            
            # return final estimates of the rates
            final_rate = nu[:,-1]

            # end loop
            break
        
    else:
        print('ERROR: solution did not converge!')  
        final_rate = np.nan*np.ones(n_pops)

    # plot to see convergence
    if plot == 1:
        plt.figure()
        for i in range(0,n_pops,1):
            plt.plot(nu[i],label=('pop %d' % i))

        plt.ylabel(r'$\nu^\mathrm{mft} \mathrm{\ [spks/sec]}$',fontsize=16)
        plt.xlabel(r'$ \mathrm{iteration \ step,} n}$',fontsize=16)
        plt.legend()
        
    
    return final_rate
 


#%% COMPUTE STATIONARY RATES WITH ROOT FINDING

# DEFINE ROOT EQUATION
def fcn_root_eqs(nu_vec, \
                 tau_r, tau_m, tau_s, \
                 Vr, Vth,  nu_ext, \
                 Jab, Cab, Jab_ext, Cab_ext, ext_variance):

    '''
    define the root equation
    '''
        
    # total number of populations
    n_pops = np.size(nu_vec)
    
    # compute mean and variance               
    Mu = fcn_compute_Mu(nu_vec, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m)
    Sigma2 = fcn_compute_Sigma2(nu_vec, nu_ext, Jab, Cab, Jab_ext, Cab_ext, tau_m, ext_variance)

    # compute standard deviation
    sigma = np.sqrt(Sigma2)
      
    # root equation
    F = np.ones((n_pops))*np.nan
    
    for i in range(0,n_pops):
        F[i] = nu_vec[i] - fcn_compute_rate(Vr[i], Vth[i], Mu[i], sigma[i], \
                                            tau_r[i], tau_m[i], tau_s[i])

    return F



# SOLVE ROOT EQUATION    
def fcn_MFT_rate_roots(nu_vec_in, nu_ext, \
                       Cab, Jab, Jab_ext, Cab_ext, ext_variance, \
                       tau_r, tau_m, tau_s, Vr, Vth):
    
    '''
    use root solver to solve self-consistent rate equations
    setting jac = False, but could also try jac = None
    '''
    
    # solve self-consistent equations
    sol = optimize.root(fcn_root_eqs, nu_vec_in, \
                        args=(tau_r, tau_m, tau_s, Vr, Vth, nu_ext, \
                              Jab, Cab, Jab_ext, Cab_ext, ext_variance),\
                        jac=False, method='hybr',
                        tol=1e-12,options={'xtol':1e-12})
        
        
    # return solution    
    return sol
