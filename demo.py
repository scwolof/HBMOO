
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(1234)

from reggie import make_gp  # GP package
from pybo import solvers    # BO package (for multi-start gradient descent)

from acquisition_functions import EHVI
from test_problems import FonzecaFleming


"""
http://oco-carbon.com/metrics/find-pareto-frontiers-in-python/

Method to take two equally-sized lists and return just the elements which lie 
on the Pareto frontier, sorted into order.
Default behaviour is to find the maximum for both X and Y, but the option is
available to specify maxX = False or maxY = False to find the minimum for either
or both of the parameters.
"""
def pareto2d (Y, maxX=False, maxY=False, return_indices=False):
    # Sort the list in either ascending or descending order of X
    if return_indices:
        myList = sorted([[Y[i,0], Y[i,1], i] for i in range(len(Y[:,0]))], reverse=maxX)
    else:
        myList = sorted([[Y[i,0], Y[i,1]] for i in range(len(Y[:,0]))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y...
                p_front.append(pair) # ... and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y...
                p_front.append(pair) # ... and add them to the Pareto frontier
    P = np.array(p_front)
    if return_indices: 
        return P[:,:2], P[:,2].astype(int)
    return P



"""
Initialise test problem
"""
func   = FonzecaFleming()    # Model object
bounds = func.bounds         # Bounds for inputs
r      = func.r              # Reference point for computing hypervolume

# Deterministic/analytic function
def detf (x, grad=False):
    return func.f1(x) if not grad else [ func.f1(x), func.df1(x) ]


"""
Initial measurement points
"""
N = 20
X = np.random.uniform(bounds[:,0], bounds[:,1], size=(N,2))
Y = np.array([ [func.f1(x), func.f2(x)] for x in X ])
P = pareto2d(Y)

P_init = P.copy()


"""
GP surrogate
"""
sn2  = 0.001               # Noise variance
rho  = 5e-07               # Signal variance
ell  = [1.] * len(bounds)  # Lengthscales
mean = 1.                  # Mean

model = make_gp(sn2, rho, ell, mean)  # Create GP surrogate
model.add_data(X, Y[:,1])  # Add data
#model.optimize()           # Optimise hyperparameters


"""
Multi-objective optimisation
"""
exp_iter = 20
for n in range (exp_iter):
    print('Additional experiment {:d}/{:d}'.format(n+1, exp_iter))
    # Acquisition function
    acqfunc = EHVI(P, r, model, detf)
    # Choose next point to evaluate
    xnext,_ = solvers.solve_lbfgs(acqfunc, bounds)

    # Make 'observation'
    ycnext = func.f1(xnext)   # Deterministic
    yfnext = func.f2(xnext)   # Black-box

    # record data
    X = np.vstack((X,xnext))
    Y = np.vstack((Y,np.array([ycnext,yfnext])))
    P = pareto2d(np.array(Y),maxX=False,maxY=False)

    # Update the model
    model.add_data(xnext,yfnext)
    #model.optimize()


plt.scatter(P[:,0], P[:,1], c='b', marker='o')
plt.scatter(P_init[:,0], P_init[:,1], c='r', marker='x')
plt.show()
