import csv
import numpy as np
import matplotlib.pyplot as plt
from lib_kalman_predictor_v0 import linear_kalman_filter

Kalman = linear_kalman_filter()

#**************************************************************************************************
#**************************************************************************************************
#**************************************************************************************************
def get_data(path,file_csv):
	T = []
	Sgamma = []
	Mgamma = []
	with open(path+file_csv, 'r') as datafile:
		ploting = csv.reader(datafile, delimiter=',') #\t
		for ROWS in ploting:
			T.append(float(ROWS[0]))
			Sgamma.append(float(ROWS[1]))
			Mgamma.append(float(ROWS[2]))
		N = len(T)
		T = np.reshape(np.array(T).T,(N,1))
		Sgamma = np.reshape(np.array(Sgamma).T,(N,1))
		Mgamma = np.reshape(np.array(Mgamma).T,(N,1))
	return T, Sgamma, Mgamma
#**************************************************************************************************
# Calculo de la matriz de transicion de estados
def F_calc(a,h):
	F11 = 1.0
	F12 = (1.0/a)*(1.0-np.exp(-a*h))
	F21 = 0.0
	F22 = np.exp(-a*h)
	F = np.array([[F11,F12],[F21,F22]])
	return F
#**************************************************************************************************
# Calculo de la matriz Bd
def Bd_calc(a,b,h):
	Bd1 = (b/a)*(h-(1.0/a)*(1.0-np.exp(-a*h)))
	Bd2 = (b/a)*(1.0-np.exp(-a*h))
	Bd = np.array([[Bd1],[Bd2]])
	return Bd
#**************************************************************************************************
# Calculo de la matriz de ruido Q
def Q_calc(sigma_Mgamma,h):
	Q = (sigma_Mgamma**2)*np.array([[1.0, 1.0/h],[1.0/h, 1.0/(h**2)]])
	return Q
#**************************************************************************************************
#**************************************************************************************************
#**************************************************************************************************
#							PARAMETROS INICIALES
path = '/home/sherlock2204f/Kalman_Filter_Series/Dual_Kalman_Predictor/'
file_csv = 'servo.csv'

# Parametros del filtro
h = 0.01									# Periodo de muestreo [s]
sigma_Mgamma = 0.009			# Des. Est. de la medicion de la posicion del servomotor
# Parametros del modelo
a = 5.0
c = 7.0  
b = 2.0
d = 1.0
# Matrices para el filtro de Kalman
F = F_calc(a,h)
Bd = Bd_calc(a,b,h)
Q = Q_calc(sigma_Mgamma,h)
R = np.array([[sigma_Mgamma**2]]) # Si es un escalar, definirlo como matriz de 1x1
H = np.array([[1.0, 0.0]])
T,S_gamma,M_gamma = get_data(path,file_csv)		# Vector de las mediciones realizadas
T = T-T[0] # Para medir desde el segundo cero
#*********************************INICIO DEL CICLO*********************************
# Inicializacion
X0 = np.array([0.0,0.0])		# Estado inicial
X0 = np.reshape(X0,(2,1))
P0 = np.diag((0.1,0.1))
# Algoritmo de Kalman
i = 0
Xk = []
N,_ = X0.shape 
Nd,m = S_gamma.shape
_,l = M_gamma.shape

# P. Pred
DM_hat = np.zeros((Nd,1))
M_hat = np.zeros((Nd,1))
a_hat = np.zeros((Nd,1))
c_hat = np.zeros((Nd,1))
b_hat = np.zeros((Nd,1))
d_hat = np.zeros((Nd,1))
GAMMA = [200.0,100.0,150.0,75.0]	# gamma1,...,gamma4
ALPHA = [100.0, 49.75]						#alpha1, alpha2 
#Par_q0 = [50.0, 50.0, 10.0, 1.0]	# a,c,b,d
#Par_q0 = [11.8, 1.2, 41.2, -2.6]	
Par_q0 = [a, c, b, d]	#a10,a20,b0,c0,d0

Wd = np.zeros((N,1))

for M in M_gamma:
	zn = np.reshape(M,(l,1))
	un = np.reshape(S_gamma[i,:],(m,1))
	# Extrapolacion
	Xn1n = Kalman.states_ext(F,Bd,Wd,X0,un)
	Pn1n = Kalman.covariance_ext(F,P0,Q)
	# Actualizacion
	Knn = Kalman.kalmanG_act(Pn1n,H,R)
	Xnn = Kalman.states_act(Knn,H,Xn1n,zn)
	Pnn = Kalman.covarianze_act(Pn1n,H,Knn,R,N)
	Xk.append(Xnn)

	# Predictor
	V = [S_gamma[i],Xk[i][0],Xk[i][1]]
	a, c, b, d, DMh, Mh = Kalman.predictor(Par_q0,V,ALPHA,GAMMA,DM_hat[i-1],M_hat[i-1],h)
	a_hat[i] = a
	c_hat[i] = c
	b_hat[i] = b
	d_hat[i] = d
	M_hat[i] = Mh
	DM_hat[i] = DMh
	Par_q0 = [a, c, b, d]	#a10,a20,b0,c0,d0
	"""
	# Actualizacion del modelo
	#if (i>4000):
	F = F_calc(a,h)
	Bd = Bd_calc(a,b,h)
	if (abs(DMh[0])>0.0): sDMh = DMh[0]/abs(DMh[0])
	else: sDMh = 0.0
	w = d-c*sDMh
	Wd = Bd_calc(a,w,h)
	"""
	# Realimentacion del algoritmo
	X0 = Xnn
	P0 = Pnn
	i = i+1
Xk = np.array(Xk)
print('i ',i)
print('a c b d ',a_hat[-1],c_hat[-1],b_hat[-1],d_hat[-1])
"""
# Guardado de datos
states = np.concatenate((T,S_gamma),axis=1)
states = np.concatenate((states,Xk[:,0]),axis=1)
states = np.concatenate((states,Xk[:,1]),axis=1)
np.save('states_ex.npy', states)
print(states.shape)
"""

# Graficas
plot1 = plt.figure(1)
plt.plot(T,0.4*S_gamma,'k.',T,M_gamma,'r-',T,Xk[:,0],'bo',T,M_hat,'g-')
plt.xlabel('[s]')
plt.ylabel('M_gamma')
plt.legend(['S control','mediciones','actualizaciones','estimaciones'])
plt.grid()

plot2 = plt.figure(2)
plt.plot(T,Xk[:,1],'b',T,DM_hat,'g')
plt.xlabel('[s]')
plt.legend(['DM_gamma','DM_hat'])
plt.grid()

plot3 = plt.figure(3)
plt.plot(T,a_hat,'r',T,c_hat,'b',T,b_hat,'g',T,d_hat,'m')
plt.xlabel('t [s]')
plt.legend(["a_hat","c_hat","b_hat","d_hat"])
plt.grid()

plt.show()
