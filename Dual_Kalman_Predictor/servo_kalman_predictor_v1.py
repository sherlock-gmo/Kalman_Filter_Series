import csv
import numpy as np
import matplotlib.pyplot as plt
from lib_kalman_predictor_v1 import linear_kalman_filter

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
def F_calc(a1,a2,h,complex_f):
	if (complex_f==True):
		c = -a1/2.0
		k = np.sqrt(a2-(a1**2/4.0))
		F11 = np.exp(c*h)*(np.cos(k*h)-(c/k)*np.sin(k*h))
		F12 = (1.0/k)*np.exp(c*h)*np.sin(k*h)
		F21 = (-a2/k)*np.exp(c*h)*np.sin(k*h)
		F22 = np.exp(c*h)*(np.cos(k*h)+(c/k)*np.sin(k*h))
		F = np.array([[F11,F12],[F21,F22]])
	else:
		p1 = (-a1+np.sqrt((a1**2)-4*a2))/2.0
		p2 = (-a1-np.sqrt((a1**2)-4*a2))/2.0
		A = 1.0/(p2-p1)
		Ap = -(p1)/(p2-p1)
		F11 = (Ap+a1*A)*np.exp(-p1*h)+(1.0-Ap-a1*A)*np.exp(-p2*h)
		F12 = A*(np.exp(-p1*h)-np.exp(-p2*h))
		F21 = (-a2*A)*(np.exp(-p1*h)-np.exp(-p2*h))
		F22 = Ap*np.exp(-p1*h)+(1.0-Ap)*np.exp(-p2*h)
		F = np.array([[F11,F12],[F21,F22]])
	return F
#**************************************************************************************************
# Calculo de la matriz Bd/Wd
def BWd_calc(a1,a2,bw,h,complex_f):
	if (complex_f==True):
		c = -a1/2.0
		k = np.sqrt(a2-(a1**2/4.0))
		BWd1 = (bw/(k*(c**2+k**2)))*(np.exp(c*h)*(c*np.sin(k*h)-k*np.cos(k*h))+k)
		BWd2 = (bw/(c**2+k**2))*(np.exp(c*h)*(k*np.sin(k*h)+c*np.cos(k*h))-c)+((bw*c)/(k*(c**2+k**2)))*(np.exp(c*h)*(c*np.sin(k*h)-k*np.cos(k*h))+k)
		BWd = np.array([[BWd1],[BWd2]])
	else:
		p1 = (-a1+np.sqrt((a1**2)-4*a2))/2.0
		p2 = (-a1-np.sqrt((a1**2)-4*a2))/2.0
		A = 1.0/(p2-p1)
		Ap = -(p1)/(p2-p1)
		BWd1 = bw*A*(((np.exp(-p2*h)-1.0)/p2)-((np.exp(-p1*h)-1.0)/p1))
		BWd2 = bw*Ap*((1.0-np.exp(-p1*h))/p1)+bw*(1.0-Ap)*((1.0-np.exp(-p2*h))/p2)
		BWd = np.array([[BWd1],[BWd2]])
	return BWd
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
a1 = 5.0 #5.0 
a2 = 5.0 #7.0 
b = 5.0 #2.0
c = 1.0 #1.0
d = 1.0 #1.0
w = 0.0
# Matrices para el filtro de Kalman
k2 = a2-(a1**2/4.0)
if (k2>0): complex_f = True
else: complex_f = False
F = F_calc(a1,a2,h,complex_f)
Bd = BWd_calc(a1,a2,b,h,complex_f)
Wd = BWd_calc(a1,a2,w,h,complex_f)
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

# Parametros del Predictor
DM_hat = np.zeros((Nd,1))
M_hat = np.zeros((Nd,1))
a1_hat = np.zeros((Nd,1))
a2_hat = np.zeros((Nd,1))
b_hat = np.zeros((Nd,1))
c_hat = np.zeros((Nd,1))
d_hat = np.zeros((Nd,1))
E = np.zeros((Nd,1))
GAMMA = [20.0,18.0,18.0,5.0,3.5] #[12.0,12.0,50.0,5.0,1.5] # gamma1,...,gamma5
ALPHA = [150, 25] #[150, 25] 						#alpha1, alpha2 
#Par_q0 = [100.0, 100.0, 50.0, 50.0, 1.0]	#a10,a20,b0,c0,d0
Par_q0 = [a1, a2, b, c, d]	

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
	a1, a2, b, c, d, DMh, Mh = Kalman.predictor(Par_q0,V,ALPHA,GAMMA,DM_hat[i-1],M_hat[i-1],h)
	a1_hat[i] = a1
	a2_hat[i] = a2
	b_hat[i] = b
	c_hat[i] = c
	d_hat[i] = d
	M_hat[i] = Mh
	DM_hat[i] = DMh
	E[i] = E[i-1]+(Xk[i][0]-Mh)**2
	Par_q0 = [a1, a2, b, c, d]	#a10,a20,b0,c0,d0

	# Actualizacion del modelo
	#w = d-c*Xk[i][1][0]/abs(Xk[i][1][0])
	#w = d-c*DMh[0]/abs(DMh[0])
	k2 = a2-(a1**2/4.0)
	if (k2>0): complex_f = True
	else: complex_f = False
	F = F_calc(a1,a2,h,complex_f)
	Bd = BWd_calc(a1,a2,b,h,complex_f)
	#Wd = BWd_calc(a1,a2,w,h,complex_f)

	# Realimentacion del algoritmo
	X0 = Xnn
	P0 = Pnn
	i = i+1
	
Xk = np.array(Xk)
print('i ',i)
print('a1 a2 b c d ',a1_hat[-1],a2_hat[-1],b_hat[-1],c_hat[-1],d_hat[-1])

#	Guardado de datos
states = np.concatenate((T,S_gamma),axis=1)
states = np.concatenate((states,Xk[:,0]),axis=1)
states = np.concatenate((states,Xk[:,1]),axis=1)
np.save('states_ex.npy', states)
np.save('Par_q0.npy', np.array(Par_q0))

print(states.shape)


# Graficas
plot1 = plt.figure(1)
plt.plot(T,0.4*S_gamma,'k.',T,M_gamma,'r-',T,Xk[:,0],'b.',T,M_hat,'g-')
plt.xlabel('[s]')
plt.ylabel('M_gamma')
plt.legend(['S control','mediciones','actualizaciones','estimaciones'])
plt.grid()
"""
plot2 = plt.figure(2)
plt.plot(T,Xk[:,1],'b',T,DM_hat,'g')
plt.xlabel('[s]')
plt.legend(['DM_gamma','DM_hat'])
plt.grid()
"""
plot3 = plt.figure(3)
plt.plot(T,a1_hat,'r',T,a2_hat,'g',T,b_hat,'m',T,c_hat,'b',T,d_hat,'y')
plt.xlabel('t [s]')
plt.legend(["a1_hat","a2_hat","b_hat","c_hat","d_hat"])
plt.grid()

plot4 = plt.figure(4)
plt.plot(T,E,'b')
plt.xlabel('t [s]')
plt.legend(["IEC"])
plt.grid()

plt.show()
