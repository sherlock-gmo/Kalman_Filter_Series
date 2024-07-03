import numpy as np
from numpy.linalg import inv

class linear_kalman_filter():
	def __init__(self):
		self.P_flag = True # Act. de covarianza: True=completa//False=simple
	#------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------
	#										ESTIMACION DE ESTADOS
	# Extrapolacion de estados
	def states_ext(self,F,B,Xnn,u):
		x1 = np.matmul(B,u)
		Xn1n = np.matmul(F,Xnn)+x1
		return Xn1n
	# Extrapolacion de la covarianza de los estados
	def X_covariance_ext(self,F,Pnn,Q):
		p1 = np.matmul(F,Pnn)
		Pn1n = np.matmul(p1,np.transpose(F))+Q
		return Pn1n
	# Actualizacion de la ganancia de Kalman de los estados
	def X_kalmanG_act(self,P,H,R):
		Ht = np.transpose(H)
		k1 = np.matmul(P,Ht)
		k2 = np.matmul(H,P)
		k3 = np.matmul(k2,Ht)+R
		k4 = inv(k3)
		Kn = np.matmul(k1,k4)
		return Kn
	# Actualizacion de estados
	def states_act(self,Kn,H,X,z):
		x1 = np.matmul(H,X)
		x2 = np.matmul(Kn,z-x1) 
		Xnn = X+x2
		return Xnn
	# Actualizacion de la covarianza de los estados
	def X_covarianze_act(self,P,H,Kn,R,N):
		# N = Numero de estados
		I = np.eye(N)
		p1 = I-np.matmul(Kn,H)
		p2 = np.matmul(p1,P)
		if (self.P_flag==False): Pnn = p2
		else:
			p3 = I-np.matmul(Kn,H)
			p4 = np.transpose(p3)
			p5 = np.matmul(p2,p4)
			p6 = np.matmul(Kn,R)
			p7 = np.matmul(p6,np.transpose(Kn))
			Pnn = p5+p7
		return Pnn
	# Algoritmo del filtro de Kalman para un conjunto de mediciones 
	def X_kalman_offline(self,X0,P0,F,B,u,H,R,Q,Y):
		Xk = []
		i = 0
		# Ajuste de tamanos
		N,_ = X0.shape 
		_,m = u.shape
		_,l = Y.shape
		for y in Y:
			yn = np.reshape(y,(l,1))
			un = np.reshape(u[i,:],(m,1))
			# Extrapolacion
			Xn1n = self.states_ext(F,B,X0,un)
			Pn1n = self.X_covariance_ext(F,P0,Q)
			# Actualizacion
			Knn = self.X_kalmanG_act(Pn1n,H,R)
			Xnn = self.states_act(Knn,H,Xn1n,yn)
			Pnn = self.X_covarianze_act(Pn1n,H,Knn,R,N)
			# Realimentacion del algoritmo
			X0 = Xnn
			P0 = Pnn
			i = i+1
			Xk.append(Xnn)
		return np.array(Xk)
	#------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------
	#										ESTIMACION DE PARAMETROS
	# Extrapolacion de parametros
	def parameters_ext(self,hnn):
		hn1n = hnn
		return hn1n
	# Extrapolacion de la covarianza de los parametros
	def P_covariance_ext(self,Pnn,lambda0):
		Q = ((1.0/lambda0)-1.0)*Pnn
		Pn1n = Pnn+Q
		return Pn1n
	# Actualizacion de la ganancia de Kalman de los parametros
	def P_kalmanG_act(self,P,H,eps):
		Ht = np.transpose(H)
		k1 = np.matmul(P,Ht)
		k2 = np.matmul(H,P)
		k3 = np.matmul(k2,Ht)
		N,_ = k3.shape
		R = eps*np.eye(N)
		k4 = inv(k3+R)
		Kn = np.matmul(k1,k4)
		return Kn
	# Actualizacion de parametros
	def parameters_act(self,h,Kn,H,d,Xe):
		x1 = np.matmul(H,Xe)
		x2 = np.matmul(Kn,d-x1) 
		hn = h+x2
		return hn
	# Actualizacion de la covarianza de los parametros
	def P_covarianze_act(self,P,H,Kn):
		p1 = np.matmul(Kn,H)
		N,_ = p1.shape
		I = np.eye(N)
		Pn = np.matmul(I-p1,P)
		return Pn
	# Algoritmo del filtro de Kalman para un conjunto de mediciones 
	def P_kalman_offline(self,D,Xe,h0,P0,H0):
		hk = []
		i = 0
		lambda0 = 0.95
		eps = 0.5
		_,n = Xe.shape
		_,l = D.shape
		for d in D:
			# Ajuste de tamanos
			dn = np.reshape(d,(l,1))
			xn = np.reshape(Xe[i,:],(n,1))
			# Extrapolacion
			hn1n = self.parameters_ext(h0)
			Pn1n = self.P_covariance_ext(P0,lambda0)
			# Actualizacion
			Knn = self.P_kalmanG_act(Pn1n,H0,eps)
			hnn = self.parameters_act(hn1n,Knn,H0,dn,xn)
			Pnn = self.P_covarianze_act(Pn1n,H0,Knn)
			# Realimentacion del algoritmo
			H0 = np.transpose(hnn)
			h0 = hnn
			P0 = Pnn
			i = i+1
			hk.append(hnn)
		return np.array(hk)
	#------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------
	#------------------------------------------------------------------------------------------

		
		
		
		
		
		
		
		
		
		
