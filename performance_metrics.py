
import numpy as np

def sqEuclid (p,P): 
    """
    Returns the squared minimum Euclidean distance between
    point p and any point in the set P
    """
    return np.sqrt(np.min(np.sum((p-P)**2,axis=1)))

def GD (Pa,Ptrue):
    """
    GD - Generational distance
    Computes average distance from points in approximated 
    Pareto frontier set Pa to true frontier Ptrue
    """
    d2 = np.array([sqEuclid(p,Ptrue) for p in Pa])
    return np.sqrt(np.sum(d2))/len(S)

def MPFE (S,Ptrue):
    """
    MPFE - Maximum Pareto Front Error
    Computes maximum distance from the true Pareto frontier Ptrue
    to any point in the approximated frontier set Pa
    """
    d2 = np.array([sqEuclid(p,Pa) for p in Ptrue])
    return np.sqrt(np.max(d2))
 
 
def Vol (P,r):
    """
    Computes area of region bounded by r and the points in P
    """
    h1 = max(np.sum(P[:,0]<r[0,0]),np.sum(P[:,1]<r[1,0]))
    h2 = min(np.sum(P[:,0]<=r[0,1]),np.sum(P[:,1]<=r[1,1]))
    Pt = P.copy()[h1:h2]
    return np.sum( (r[1,1]-Pt[:,1]) * (np.append(Pt[1:,0],r[0,1])-Pt[:,0]) )

def VR (P,r,PtrueVol):
    """
    Ratio of true area covered by approximated Pareto front P
    """
    return -np.log(1.-Vol(P,r)/PtrueVol)
