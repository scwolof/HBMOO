
import numpy as np 
from scipy.stats import norm as ssn


class EHVI:
	"""
	EXPECTED HYPER-VOLUME IMPROVEMENT

	Inputs:
		P 		Pareto front. P.shape = [np,2]
		r 		Boundaries. r = np.array([[min(f_det),max(f_det)],[min(f_prob),max(f_prob)]])
		model	GP model (see reggie package)
		detf	Deterministic function. Must take array inputs, and return gradients if grad=True (see use below).

	Outputs:
		EI 		Expected improvement
		dEI 	(if grad=True) gradients of expected improvement w.r.t. input X  

	(C) 2018 by Simon Olofsson
	
	"""

	def __init__ (self,P,r,model,detf):
		# Pareto front
		self.P = P.copy()
		# Function boundaries
		self.r = r
		# Probabilistic model
		self.model = model.copy()
		# Deterministic function
		self.detf = detf
		# Extended pareto points
		self.pe = np.vstack((np.array([r[0,0],r[1,1]]),P.copy(),np.array([r[0,1],r[1,0]])))
		# Volumes
		self.V = np.cumsum(np.append(np.append(0,(self.pe[:-2,1]-self.P[:,1])*(self.r[0,1]-self.P[:,0])),0))

	def __call__ (self,X,grad=False):
		X = X.copy()
		if X.ndim == 1: X = X[None,:]
		leng = len(X)

		# Compute yhat and variance
		post = self.model.predict(X,grad=grad)
		if grad: 
			c,dc = self.detf(X,grad=True)
			dh = np.zeros(X.shape)
		else: 
			c = self.detf(X,grad=False)

		mu,s2 = post[:2]
		yhat = np.c_[c,mu]
		
		h = np.zeros(leng)
		for m in range(leng):
			ht = self.ehvisum(yhat[m],s2[m],grad=grad)
			if not grad: 
				h[m] = ht
			else:
				h[m] = ht[0]
				dh[m] = np.dot(ht[1],np.vstack((dc[m],post[2][m]))) + ht[2]*post[3][m]

		if not grad: return h
		return h, dh

	def ehvisum (self,mu,s2,grad=False):

		# Standard deviation
		s = np.sqrt(s2)
		# Distribution with mean and standard deviation
		dist = ssn(mu[1],s)

		# Lower index
		h1 = np.sum(self.P[:,0]<mu[0])
		# Range of j's
		j = np.array(range(h1,len(self.P)+1))
		# Upper indices
		h2 = j+1
		cdf1 = dist.cdf(self.pe[j,1]); cdf2 = dist.cdf(self.pe[h2,1])
		cdf = cdf1-cdf2
		pdf1 = dist.pdf(self.pe[j,1]); pdf2 = dist.pdf(self.pe[h2,1])
		pdf = pdf1-pdf2
		# Expected hyper-volume improvement
		sum1 = (self.V[h1] - self.V[h2] + self.pe[h1,1]*(self.r[0,1]-mu[0]) - self.pe[h2,1]*(self.r[0,1]-self.pe[h2,0]) + mu[1]*(mu[0]-self.pe[h2,0]))
		EI = np.sum(sum1*cdf - (mu[0]-self.pe[h2,0])*s2*pdf)
		# Gradients
		if not grad: return EI
		t0 = pdf1*(self.pe[j,1]-mu[1])-pdf2*(self.pe[j+1,1]-mu[1])
		t1 = pdf1*((self.pe[j,1]-mu[1])**2/(2*s2)-0.5) - pdf2*((self.pe[j+1,1]-mu[1])**2/(2*s2)-0.5)
		dmu0 = np.sum((mu[1]-self.pe[h1,1])*cdf - s2*pdf)
		dmu1 = np.sum((mu[0]-self.pe[h2,0])*cdf - sum1*pdf - (mu[0]-self.pe[h2,0])*t0)
		ds2 = np.sum(-0.5*sum1*t0/s2 - (mu[0]-self.pe[h2,0])*(pdf+t1))
		return EI, np.array([dmu0, dmu1]), ds2




class EMmI:
	"""
	EXPECTED MAXIMIN FITNESS

	Inputs:
		P 		Pareto front. P.shape = [np,2]
		model	GP model (see reggie package)
		detf	Deterministic function. Must take array inputs, and return gradients if grad=True (see use below).

	Outputs:
		EI 		Expected maximin fitness
		dEI 	(if grad=True) gradients of expected maximin fitness w.r.t. input X  

	(C) 2018 by Simon Olofsson
	
	"""

	def __init__ (self,P,model,detf):
		# Pareto front
		self.P = P.copy()
		# Probabilistic model
		self.model = model.copy()
		# Deterministic function
		self.detf = detf

	def __call__ (self,X,grad=False):
		X = X.copy()
		if X.ndim == 1: X = X[None,:]
		leng = len(X)

		# Compute yhat and variance
		post = self.model.predict(X,grad=grad)
		if grad: 
			c,dc = self.detf(X,grad=True)
			dh = np.zeros(X.shape)
		else: 
			c = self.detf(X,grad=False)

		mu,s2 = post[:2]
		yhat = np.c_[c,mu]
		
		h = np.zeros(leng)
		for m in range(leng):
			ht = self.emmisum(yhat[m],s2[m],grad=grad)
			if not grad: 
				h[m] = ht
			else:
				h[m] = ht[0]
				dh[m] = np.dot(ht[1],np.vstack((dc[m],post[2][m]))) + ht[2]*post[3][m]

		if not grad: return h
		return h, dh

	def emmisum (self,mu,s2,grad=False):

		# Standard deviation
		s = np.sqrt(s2)
		# Distribution with mean and standard deviation
		dist = ssn(mu[1],s)

		# Get points on Pareto front, for axis i, shifted by add_to_j
		def p (i,add_to_j): 
			if add_to_j < 0: return np.append(float('inf'),self.P[:-1,i])
			elif add_to_j > 0: return np.append(self.P[1:,i],float('inf'))
			else: return self.P[:,i].copy()		

		# sum Int(1,:)
		def Int1 (): 
			# Define bounds
			bounds = np.c_[ mu[0]+p(1,0)-p(0,0), mu[0]+p(1,-1)-p(0,0) ]
			cdf = dist.cdf(bounds[:,1]) - dist.cdf(bounds[:,0])
			# Delta function 
			delta = (mu[0] < p(0,0)).astype(int)
			# Sum elements for which mu[0] < p(0,j)
			int1 = np.dot(delta,(p(0,0)-mu[0])*cdf)
			if not grad: return int1
			# Gradients
			pdf0 = dist.pdf(bounds[:,0])
			pdf1 = dist.pdf(bounds[:,1])
			pdf = (p(0,0)-mu[0])*(pdf1-pdf0)
			dmu = np.dot(delta, np.c_[-cdf+pdf, -pdf])
			ds2 = np.dot(delta,-0.5*(p(0,0)-mu[0])*(pdf1*(np.append(0,bounds[1:,1])-mu[1])/s2 - 
											   		pdf0*(bounds[:,0]-mu[1])/s2))
			return int1, dmu, ds2

		# sum Int(2,:)
		def Int2 ():
			# Define bounds
			bounds = np.c_[ mu[0]+p(1,0)-p(0,1), np.amin(np.c_[p(1,0), mu[0]+p(1,0)-p(0,0)], axis=1) ]
			cdf = dist.cdf(bounds[:,1]) - dist.cdf(bounds[:,0])
			pdf0 = dist.pdf(bounds[:,0])
			pdf1 = dist.pdf(bounds[:,1])
			pdf = pdf1-pdf0
			# Delta function 
			delta = (mu[0] < p(0,1)).astype(int)
			# Sum elements for which mu[0] < p(0,j+1)
			int2 = np.dot(delta, (p(1,0)-mu[1])*cdf + s2*pdf)
			if not grad: return int2
			# Gradients
			t0 = pdf1*(bounds[:,1]-mu[1]) - pdf0*(np.append(bounds[:-1,0],0)-mu[1])
			t1 = pdf1*((bounds[:,1]-mu[1])**2/(2*s2)-0.5) - pdf0*((np.append(bounds[:-1,0],0)-mu[1])**2/(2*s2)-0.5)
			dmu = np.dot(delta, np.c_[ (p(1,0)-mu[1])*pdf-t0, -(cdf+(p(1,0)-mu[1])*pdf)+t0 ])
			ds2 = np.dot(delta,-0.5*(p(1,0)-mu[1])*t0/s2 + pdf+t1)

			return int2, dmu, ds2

		i1 = Int1(); i2 = Int2()
		if not grad: return i1+i2

		emmi = i1[0] + i2[0]
		dmu = i1[1] + i2[1]
		ds = i1[2] + i2[2]
		return emmi, dmu, ds






