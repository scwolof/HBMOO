
import numpy as np 
from scipy.stats import norm as ssn

from pdb import set_trace as st


class EHVI:
    """
    EXPECTED HYPER-VOLUME IMPROVEMENT

    Inputs:
        P       Pareto front. P.shape = [n, 2]
        r       Boundaries. r = np.array([ [min(f_det),  max(f_det)],
                                           [min(f_prob), max(f_prob)] ])
        model   GP model (see reggie package)
        detf    Deterministic function. Must take array inputs, and return
                gradients if grad=True (see use below).

    Outputs:
        EI      Expected improvement
        dEI     (if grad=True) gradients of EI w.r.t. input X  

    (C) 2018 by Simon Olofsson
    
    """
    def __init__ (self,P,r,model,detf):
        self.P     = P
        self.r     = r
        self.model = model.copy()
        self.detf  = detf
    @property
    def pe (self):
        # Extended pareto points
        p0 = np.array([self.r[0,0],self.r[1,1]])
        p1 = np.array([self.r[0,1],self.r[1,0]])
        return np.vstack((p0,self.P,p1))
    @property
    def V (self):   
        # Area covered by first i points on Pareto frontier
        V0 = 0
        Vs = (self.pe[:-2,1]-self.P[:,1]) * (self.r[0,1]-self.P[:,0])
        return np.cumsum( np.concatenate(([0],Vs,[0])) )

    def __call__ (self,X,grad=False):
        X = X.copy() if X.ndim > 1 else X[None,:]

        # Compute yhat and variance
        post  = self.model.predict(X,grad=grad)
        mu,s2 = post[:2]
        f     = self.detf(X,grad=grad)
        if grad: 
            f,df = f
            dh   = np.zeros(X.shape)
            dmu  = post[2]
            ds2  = post[3]
        
        h = np.zeros(len(X))
        for m in range(len(X)):
            ht = self.ehvisum(f[m],mu[m],s2[m],grad=grad)
            if not grad: 
                h[m] = ht
            else:
                h[m]  = ht[0]
                dh[m] = ht[1]*df[m] + ht[2]*dmu[m] + ht[3]*ds2[m]

        return h if not grad else [h,dh]

    def ehvisum (self,f,mu,s2,grad=False):
        dist = ssn(mu,np.sqrt(s2)) # distribution

        h1 = np.sum( self.P[:,0] < f )      # Lower index
        j  = np.arange(h1, len(self.P)+1)   # Range of j's
        h2 = j + 1                          # Upper indices

        pe = self.pe

        cdf1 = dist.cdf( pe[ j, 1] )
        cdf2 = dist.cdf( pe[h2, 1] )
        cdf  = cdf1 - cdf2

        pdf1 = dist.pdf( pe[ j, 1] )
        pdf2 = dist.pdf( pe[h2, 1] )
        pdf  = pdf1 - pdf2

        fp2  = f-pe[h2,0]
        # Expected hyper-volume improvement
        V  = self.V
        r  = self.r
        sum1 = (V[h1] - V[h2] + mu * fp2
                + pe[h1,1] * (r[0,1]-f) \
                - pe[h2,1] * (r[0,1]-pe[h2,0]))
        EI = np.sum( sum1*cdf - fp2*s2*pdf )
        if not grad: return EI

        # Gradients
        mp1 = pe[j,  1] - mu
        mp2 = pe[j+1,1] - mu
        t0  = pdf1*mp1 - pdf2*mp2
        t1  = pdf1*(mp1**2/(2*s2)-0.5) - pdf2*(mp2**2/(2*s2)-0.5)
        df  = np.sum( (mu-pe[h1,1])*cdf - s2*pdf )
        dmu = np.sum( fp2*cdf - sum1*pdf - fp2*t0 )
        ds2 = np.sum( -0.5*sum1*t0/s2 - fp2*(pdf+t1) )
        return EI, df, dmu, ds2




class EMmI:
    """
    EXPECTED MAXIMIN FITNESS

    Inputs:
        P       Pareto front. P.shape = [np,2]
        model   GP model (see reggie package)
        detf    Deterministic function. Must take array inputs, and return 
                gradients if grad=True (see use below).

    Outputs:
        EI      Expected maximin fitness
        dEI     (if grad=True) gradients of EI w.r.t. input X

    (C) 2018 by Simon Olofsson
    
    """

    def __init__ (self,P,model,detf):
        self.P     = P
        self.model = model.copy()
        self.detf  = detf

    def __call__ (self,X,grad=False):
        X = X.copy() if X.ndim > 1 else X[None,:]

        # Compute yhat and variance
        post  = self.model.predict(X,grad=grad)
        mu,s2 = post[:2]
        f     = self.detf(X,grad=grad)
        if grad: 
            f,df = f
            dh   = np.zeros(X.shape)
            dmu  = post[2]
            ds2  = post[3]
        
        h = np.zeros(len(X))
        for m in range(len(X)):
            ht = self.emmisum(f[m],mu[m],s2[m],grad=grad)
            if not grad: 
                h[m] = ht
            else:
                h[m]  = ht[0]
                dh[m] = ht[1]*df[m] + ht[2]*dmu[m] + ht[3]*ds2[m]

        return h if not grad else [h,dh]

    def emmisum (self,f,mu,s2,grad=False):
        dist = ssn(mu,np.sqrt(s2)) # distribution

        pf0  = self.P[:,0] - f

        # sum Int(1,:)
        def Int1 ():
            p1   = np.append(np.inf, self.P[:-1,0])

            dlt  = ( pf0 >= 0 ).astype(int)
            # Define bounds
            bnds = np.c_[ self.P[:,1]-pf0, p1-pf0 ]
            
            cdf  = dlt * (dist.cdf(bnds[:,1]) - dist.cdf(bnds[:,0]))

            int1 = np.sum( pf0 * cdf)
            if not grad: return int1

            pdf0 = dlt * dist.pdf(bnds[:,0])
            pdf1 = dlt * dist.pdf(bnds[:,1])
            pdf  = pf0 * (pdf1 - pdf0)
            # Gradients
            df   =  np.sum( pdf - cdf )
            dmu  = -np.sum( pdf )
            
            pdf0 *= bnds[:,0] - mu
            pdf1 *= np.append(0,bnds[1:,1]) - mu
            ds2  = -np.sum(0.5 * pf0 * (pdf1 - pdf0)/s2)
            return int1, df, dmu, ds2

        # sum Int(2,:)
        def Int2 ():
            pf1  = np.append(self.P[1:,0], np.inf) - f
            p1   = self.P[:,1]
            pm1  = p1 - mu

            dlt  = (pf1 >= 0).astype(int)
            # Define bounds
            bnds = np.c_[ p1-pf1, np.amin(np.c_[p1,p1-pf0],axis=1) ]

            cdf  = dlt * (dist.cdf(bnds[:,1]) - dist.cdf(bnds[:,0]))

            pdf0 = dlt * dist.pdf(bnds[:,0])
            pdf1 = dlt * dist.pdf(bnds[:,1])
            pdf  = pdf1 - pdf0

            int2 = np.sum(pm1*cdf + s2*pdf)
            if not grad: return int2

            d0   = np.append(bnds[:-1,0],0) - mu
            d1   = bnds[:,1] - mu
            t0   = pdf1 * d1 - pdf0 * d0
            t1   = 0.5 * (pdf1 * d1**2 - pdf0 * d0**2)/s2 - 0.5 * pdf
            # Gradients
            df   = np.sum( pm1 * pdf - t0)
            dmu  = np.sum( cdf ) - df
            ds2  = np.sum( pdf + t1 - 0.5 * pm1 * t0/s2)

            return int2, df, dmu, ds2

        i1 = Int1(); i2 = Int2()
        if not grad: return i1+i2

        EI   = i1[0] + i2[0]
        df   = i1[1] + i2[1]
        dmu  = i1[2] + i2[2]
        ds   = i1[3] + i2[3]
        return EI, df, dmu, ds

