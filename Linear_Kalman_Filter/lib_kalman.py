import numpy as np
from numpy.linalg import inv

class linear_kalman_filter():
	def __init__(self):
		self.P_flag = True # Act. de covarianza: True=completa//False=simple
	# Extrapolacion de estados
	def states_ext(self,F,B,Xnn,u):
		x1 = np.matmul(B,u)
		Xn1n = np.matmul(F,Xnn)+x1
		return Xn1n
	# Extrapolacion de la covarianza
	def covariance_ext(self,F,Pnn,Q):
		p1 = np.matmul(F,Pnn)
		Pn1n = np.matmul(p1,np.transpose(F))+Q
		return Pn1n
	# Actualizacion de la ganancia de Kalman
	def kalmanG_act(self,P,H,R):
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
	# Actualizacion de la covarianza
	def covarianze_act(self,P,H,Kn,R,N):
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
	def kalman_offline(self,X0,P0,F,B,u,H,R,Q,Z):
		Xk = []
		i = 0
		# Ajuste de tamanos
		N,_ = X0.shape 
		_,m = u.shape
		_,l = Z.shape
		for z in Z:
			zn = np.reshape(z,(l,1))
			un = np.reshape(u[i,:],(m,1))
			# Extrapolacion
			Xn1n = self.states_ext(F,B,X0,un)
			Pn1n = self.covariance_ext(F,P0,Q)
			# Actualizacion
			Knn = self.kalmanG_act(Pn1n,H,R)
			Xnn = self.states_act(Knn,H,Xn1n,zn)
			Pnn = self.covarianze_act(Pn1n,H,Knn,R,N)
			# Realimentacion del algoritmo
			X0 = Xnn
			P0 = Pnn
			i = i+1
			Xk.append(Xnn)
		return np.array(Xk)

		
		
		
		
		
		
		
		
		
		
		
		
		
		
