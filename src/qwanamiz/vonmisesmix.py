# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:04:48 2024

@author: sambo
"""

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate and fits some periodic von Mises distribution functions. 

From https://framagit.org/fraschelle/mixture-of-von-mises-distributions

All the functions below can be called directly from the main package name. That is

```python
import vonMisesMixtures as vm

vm.density(x, mu, kapp)
vm.mixtures.density(x, mu, kappa)
```
return the same things, and the same is true for all the functions below.
"""

from typing import Tuple 

import numpy as np
from scipy.special import iv
from scipy.optimize import fsolve

def vonmises_density(x: np.array, mu: np.array, kappa: np.array) -> np.array:
    """Alias for `density` function. Kept for retro-compatibility"""
    return density(x, mu, kappa)

def density(x: np.array, mu: np.array, kappa: np.array) -> np.array:
    """
    Calculate the von Mises density for a series x (a 1D numpy.array).
    
    Input | Type | Details
    -- | -- | --
    x | a 1D numpy.array of size L |
    mu | a 1D numpy.array of size n | the mean of the von Mises distributions
    kappa | a 1D numpy.array of size n | the dispersion of the von Mises distributions
    
    Output : 
        a (L x n) numpy array, L is the length of the series, and n is the size of the array containing the parameters. Each row of the output corresponds to a density
    """    
    not_normalized_density = np.array([np.exp(kappa*np.cos(i-mu)) for i in x])
    norm = 2*np.pi*iv(0,kappa)
    _density = not_normalized_density/norm
    return _density

def pdfit(series: np.array) -> Tuple[float]:
    """
    Calculate the estimator of the mean and deviation of a sample, for a von Mises distribution
    
    Input : 
        series : a 1D numpy.array
        
    Output : 
        the estimators of the parameters mu and kappa of a von Mises distribution, in a tuple (mu, kappa)
    See https://en.wikipedia.org/wiki/Von_Mises_distribution 
    for more details on the von Mises distribution and its parameters mu and kappa.
    """
    s0 = np.mean(np.sin(series))
    c0 = np.mean(np.cos(series))
    mu = np.arctan2(s0,c0)
    var = 1-np.sqrt(s0**2+c0**2)
    k = lambda kappa: 1-iv(1,kappa)/iv(0,kappa)-var
    kappa = fsolve(k, 0.0)[0]
    return mu, kappa 

def vonmises_pdfit(series: np.array) -> Tuple[float]:
    """Alias for `pdfit` function. Kept for retro-compatibility."""
    return pdfit(series)

def mixture_pdfit(series: np.array, pi, mu, kappa, n: int=2, threshold: float=1e-3) -> np.array:
    """
    Find the parameters of a mixture of von Mises distributions, using an EM algorithm.
    
    Input | Type | Details
    -- | -- | --
    series | a 1D numpy array | represent the stochastic periodic process
    n | an int | the number of von Mises distributions in the mixture
    pi | initial values for pi (proportions of each distribution)
    mu | initial values for mu (measure of location, analogous to mean) 
    kappa | initial values for kappa (measure of concentration, analogous to 1/variance)
    threshold | a float | correspond to the euclidean distance between the old parameters and the new ones
    
    Output : a (3 x n) numpy-array, containing the probability amplitude of the distribution, 
    and the mu and kappa parameters on each line.
    """
    # initialise the parameters and the distributions
    t = pi*vonmises_density(series, mu, kappa)
    s = np.sum(t, axis=1)
    t = (t.T/s).T
    thresh = 1.0
    # calculate and update the coefficients, untill convergence
    while thresh > threshold:
        new_pi = np.mean(t, axis=0)
        new_mu = np.arctan2(np.sin(series)@t, np.cos(series)@t)      
        c = np.cos(series)@(t*np.cos(new_mu)) + np.sin(series)@(t*np.sin(new_mu))
        k = lambda kappa: (c-iv(1, kappa)/iv(0, kappa)*np.sum(t, axis=0)).reshape(n)
        new_kappa = fsolve(k, np.zeros(n))
        thresh = np.sum((pi-new_pi)**2 + (mu-new_mu)**2 + (kappa-new_kappa)**2)
        pi = new_pi
        mu = new_mu
        kappa = new_kappa
        t = pi*vonmises_density(series,mu,kappa)
        s = np.sum(t, axis=1)
        t = (t.T/s).T
    res = np.array([pi, mu, kappa])
    # in case there is no mixture, one fits the data using the estimators
    if n == 1:
        res = vonmises_pdfit(series)
        res = np.append(1.0,res)
        res = res.reshape(3,1)
    return res

def mixture_vonmises_pdfit(series: np.array, n: int=2, threshold: float=1e-3) -> np.array:
    """Alias for `mixture_pdfit` function. Kept for retro-compatibility."""
    return mixture_pdfit(series, n=n, threshold=threshold)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools to 
 - substract a linear background from a von Mises distribution, 
 - generate a histogram from a random sample, 
 - generate an artificial mixtures of von Mises distributions,
 - calculate the Hellinger distance between two probability density functions.
 
"""

import numpy as np

def histogram(series, bins=50):
    """Construct the histogram from the random series. The function associates 
    the middle of the bin coordinates to its normalized frequency.
    
    Input | Type | Details
    -- | -- | --
    series | list or numpy.array | a random sample
    bins  | int | the number of points in the outcome
        
    Output : 
        (x, y) tuple, two numpy 1D arrays with y[i] the frequency and x[i] its 
        center coordinate
    """
    histo = np.histogram(series, bins)
    histo_norm = np.array(histo[0])/np.sum(histo[0])
    coords = histo[1][:-1]+histo[1][1:]
    return coords/2, histo_norm

def least_square_periodic(x, y):
    """
    Apply the least square method for a linear fit of a sample, periodic in the
    y variable with period `2*numpy.pi`. Returns the a and b parameters of the
    fit `y = a*x + b`
    
    Input | Type | Details
    -- | -- | --
    `x` | a 1D `numpy.array` of size L |
    `y` | a 1D numpy.array of size L |
        
    Output : 
        (a, b) tuple, two float numbers with `y_fit = a*x + b`
    """
    assert len(x) == len(y), "x and y must be of same length"
    xm = np.mean(x)
    ym = np.arctan2(np.mean(np.sin(y)),np.mean(np.cos(y)))
    numer = (x-xm)*np.arctan2(np.sin(y-ym),np.cos(y-ym))
    denom = (x-xm)**2
    a = np.sum(numer)/np.sum(denom)
    # a = np.arctan2(np.sin(a),np.cos(a))
    b = ym - a*xm
    # b = np.arctan2(np.sin(b),np.cos(b))
    return a, b

def hellinger_dist(x1, x2):
    """
    Hellinger distance between the two samples distribution histogram
    of same size x1 and x2.
    See [wiki:Hellinger_distance](https://en.wikipedia.org/wiki/Hellinger_distance) for definitions.
    
    Input : x1, x2 : lists or numpy 1D arrays
    
    Output : A float number
    """
    assert len(x1) == len(x2), "The two series must have the same length to be compared"
    h = np.sqrt(x1)-np.sqrt(x2)
    h = np.sum(h**2)
    h = np.sqrt(h/2)
    return h

def generate_mixtures(p=[0.5,0.5],
                      mus=[0,3.14], 
                      kappas=[2,5], 
                      sample_size=100):
    """Generate a sample of size `sample_size` of several von Mises distributions
    (the number of distribution is `len(p)==len(mus)==len(kappas)`).
     
     Input | Type | Details
     -- | -- | --
     p | a list of floating number summing to 1 | each float number represent the relative probability of each von Mises Sub-Distribution (vMSD)
     mus | a list of floats | The parameter mu of each of the vMSD.
     kappas | a list of floats | The parameter kappa of each of the vMSD
     sample_size | an int | the number of points in the random sample
         
         
    Output : an array of size `sample_size`
    """
    assert len(p)==len(mus), "all lists must have same length"
    assert len(mus)==len(kappas), "all lists must have same length"
    sample = list()
    for _ in range(sample_size):
        s = np.random.choice(range(len(mus)),p=p)
        choice = np.random.vonmises(mus[s],kappas[s])
        sample.append(choice)
    return np.array(sample)
