
import numpy as np 

"""
Test functions found on 
https://en.wikipedia.org/wiki/Test_functions_for_optimization

Plus two additional: Sminus and Splus
"""

# CONCAVE
class FonzecaFleming:
	def __init__ (self,n=2):
		self.name = 'FonzecaFleming'
		self.n    = n
	@property
	def invn (self):
		return 1./np.sqrt(self.n)
	@property
	def bounds (self):
		return np.array([[-4.,4.]]*self.n)
	@property
	def r (self):
		return np.array([[0.,1.],[0.,1.]])

	def f1 (self,x):
		return self.f(x,1)
	def f2 (self,x): 
		return self.f(x,2)
	def f (self,x,i):
		xx = (x + (2*i-3)*self.invn)**2
		sx = np.sum(xx, axis = None if x.ndim == 1 else 1)
		return 1. - np.exp(-sx)

	# Derivatives
	def df1 (self,x):
		return self.df(x,1)
	def df2 (self,x):
		return self.df(x,2)
	def df (self,x,i):
		f  = 1. - self.f(x,i)
		f  = f if x.ndim == 1 else f[:,None]
		dx = 2.*(x + (2*i-3)*self.invn) * f
		return dx if dx.ndim > 1 else dx[None,:]

# DISCONTINUOUS
class Kursawe:
	def __init__ (self,n=3):
		self.name = 'Kursawe'
		self.n    = n
	@property
	def bounds (self):
		return np.array([[-5.,5.]]*self.n)
	@property
	def r (self):
		b1 = -10 * (self.n - 1.) - 1e-2
		b2 = -10 * (self.n - 1.) * np.exp(-np.sqrt(2)) + 1e-2
		b3 = -3.8757617 * self.n - 1e-1
		b4 =  8.6168728 * self.n + 1e-1
		return np.array([[b1,b2],[b3,b4]])

	def f1 (self,x):
		m = x.ndim
		x = x.copy() if m > 1 else x.copy()[None,:]
		xx = -10. * np.exp(-0.2*np.sqrt(x[:,:-1]**2 + x[:,1:]**2))
		sx = np.sum( xx, axis = None if m == 1 else 1 )
		return sx
	def f2 (self,x): 
		xx = np.abs(x)**0.8 + 5*np.sin(x**3)
		return np.sum( xx, axis = None if x.ndim == 1 else 1 )

	# Derivatives
	def df1 (self,x): 
		x    = x.copy() if x.ndim > 1 else x.copy()[None,:]
		sqrt = lambda x1,x2: np.sqrt( x1**2 + x2**2 )
		f    = lambda x1,x2: 2. * x1/sqrt(x1,x2) * np.exp(-0.2*sqrt(x1,x2))
		dx   = np.zeros((len(x),self.n))
		for i in range(self.n):
			if i > 0:
				dx[:,i] += f(x[:,i],x[:,i-1])
			if i < self.n-1:
				dx[:,i] += f(x[:,i],x[:,i+1])
		return dx
	def df2 (self,x):
		dx = 0.8 * np.sign(x) * np.abs(x)**(-0.2) 
		dx[np.isnan(dx)] = 0.
		return dx + 15. * np.cos(x**3) * x**2


# CONVEX
class Schaffer:
	def __init__ (self):
		self.name = 'Schaffer'
		self.n    = 1
	@property
	def bounds (self):
		return np.array([[-3.,3.]])
	@property
	def r (self):
		return np.array([[0.,3],[0.,25]])

	def f1 (self,x): 
		return x**2
	def f2 (self,x): 
		return (x-2)**2

	# Derivatives
	def df1 (self,x): 
		return 2*x
	def df2 (self,x): 
		return 2*(x-2)

# MIXED
class S:
	def __init__ (self,i):
		self.i = i 	# Decides Sminus or Splus
		self.n = 2
	@property
	def bounds (self):
		sigma = 2.
		return np.array([[0.,10.],[0.,sigma]])
	@property
	def r (self):
		sigma = 2.
		return np.array([[0.,10.],[0.,10.+sigma]])

	def f1 (self,x): 
		return x[0] if x.ndim == 1 else x[:,0]
	def f2 (self,x): 
		ns = x[1] if x.ndim == 1 else x[:,1]
		f1 = self.f1(x)
		return (10. - f1) + (2*self.i-3) * np.sin(f1) + ns

	# Derivatives
	def df1 (self,x):
		return np.array( [[1,0]]*len(x) ) 
	def df2 (self,x): 
		f1  = self.f1(x)
		dx0 = -np.ones(len(x)) + (2*self.i-3)*np.cos(f1)
		dx1 =  np.ones(len(x))
		return np.c_[ dx0, dx1 ]

class Sminus (S):
	def __init__ (self):
		S.__init__(self,1)
		self.name = 'S-'

class Splus (S):
	def __init__ (self):
		S.__init__(self,2)
		self.name = 'S+'
