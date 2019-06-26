# -*- coding: utf-8 -*-
"""Optimizers for optimizing feature detector parameters.
"""

import numpy as np


def cross_entropy_optimizer(func, xrng, pop_size=20, elite_size=10,
                            max_iters=10, mu=None, sigma=None, print_elite_costs=True):
    """Cross-entropy optimizer for minimizing cost function of parameters in the
    specified ranges.
    """

    xlow, xhigh = xrng[:, 0], xrng[:, 1]
    assert np.all(xhigh >= xlow)
    assert elite_size <= pop_size
    
    # Specify mu and sigma based on bounds if not provided
    mu = 0.5 * (xlow + xhigh) if mu is None else mu
    sigma = 0.5 * (xhigh - xlow) if sigma is None else sigma

    n_vars = xlow.size    
    alpha, beta = 0.7, 0.9
    q = 7
    
    for i in range(1, max_iters + 1):
        # Generate valid population using rejection sampling
        pop = mu + sigma * np.random.randn(pop_size, n_vars)
        pop = pop[np.logical_and(np.all(pop >= xlow.T, axis=1),
                                 np.all(pop <= xhigh.T, axis=1))]
        while pop.shape[0] < pop_size:
            pop2 = mu + sigma * np.random.randn(pop_size, n_vars)
            pop2 = pop2[np.logical_and(np.all(pop2 >= xlow.T, axis=1),
                                       np.all(pop2 <= xhigh.T, axis=1))]
            pop = np.vstack((pop, pop2))
        pop = pop[:pop_size]
        
        # Evaluate population costs
        cost = np.array([func(pop[j]) for j in range(pop_size)])
        
        # Sort population by cost and select elite members
        order = np.argsort(cost)
        elite = pop[order[:elite_size]]
        elite_cost = cost[order[:elite_size]]
        
        if print_elite_costs:
            print("{}/{} : {}".format(i, max_iters, elite_cost))
        
        # Update search distribution using elite members
        mu = alpha * np.mean(elite, axis=0) + (1 - alpha) * mu
        beta_t = beta - beta * (1 - 1 / float(i)) ** q
        sigma = beta_t * np.std(elite, axis=0) + (1 - beta_t) * sigma
        
        # Check for convergence
        if np.mean(sigma ** 2) < 1e-5:
            break

    # Return mean of elite samples as optimal value
    return mu

    