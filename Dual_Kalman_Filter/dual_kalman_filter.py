import csv
import numpy as np
import matplotlib.pyplot as plt
from lib_kalman import linear_kalman_filter

Kalman = linear_kalman_filter()
#**************************************************************************************************
#**************************************************************************************************
#**************************************************************************************************
def get_data(path,file_csv):
	T = []
	Sgamma = []
	Mgamma = []
	with open(path+file_csv, 'r') as datafile:
		ploting = csv.reader(datafile, delimiter='\t') #\t #,
		for ROWS in ploting:
			T.append(float(ROWS[0]))
			Sgamma.append(float(ROWS[3])) #3
			Mgamma.append(float(ROWS[4])) #4
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
# Calculo de la matriz Bd
def Bd_calc(a1,a2,b,h,complex_f):
	if (complex_f==True):
		c = -a1/2.0
		k = np.sqrt(a2-(a1**2/4.0))
		Bd1 = (b/(k*(c**2+k**2)))*(np.exp(c*h)*(c*np.sin(k*h)-k*np.cos(k*h))+k)
		Bd2 = (b/(c**2+k**2))*(np.exp(c*h)*(k*np.sin(k*h)+c*np.cos(k*h))-c)+((b*c)/(k*(c**2+k**2)))*(np.exp(c*h)*(c*np.sin(k*h)-k*np.cos(k*h))+k)
		Bd = np.array([[Bd1],[Bd2]])
	else:
		p1 = (-a1+np.sqrt((a1**2)-4*a2))/2.0
		p2 = (-a1-np.sqrt((a1**2)-4*a2))/2.0
		A = 1.0/(p2-p1)
		Ap = -(p1)/(p2-p1)
		Bd1 = b*A*(((np.exp(-p2*h)-1.0)/p2)-((np.exp(-p1*h)-1.0)/p1))
		Bd2 = b*Ap*((1.0-np.exp(-p1*h))/p1)+b*(1.0-Ap)*((1.0-np.exp(-p2*h))/p2)
		Bd = np.array([[Bd1],[Bd2]])
	return Bd
#**************************************************************************************************
# Calculo de la matriz de ruido Q
def Q_calc(sigma_Mgamma,h):
	Q = (sigma_Mgamma**2)*np.array([[1.0, 1.0/h],[1.0/h, 1.0/(h**2)]])
	return Q
#**************************************************************************************************
# Calculo del vector de estados extendido Xe
def Xe_calc(X,u):
	n,m,_ = X.shape
	X = np.reshape(X,(n,m))
	Xe = np.concatenate((-X,u),axis=1)
	return Xe
#**************************************************************************************************
# Calculo del vector de salidas d
def d_calc(DM_gamma):
	#fc = 47.7465
	#wc = 2*np.pi*fc
	#df0 = 0.0
	N,_ = DM_gamma.shape
	D = np.zeros((N,1))
	for i in range (0,N):
		# Derivada numerica
		dn = (DM_gamma[i]-DM_gamma[i-1])/h 
		# # Filtro pasa-altos
		#df = (1.0/(1.0+h*wc))*(wc*(DM_gamma[i]-DM_gamma[i-1])+df0) 
		D[i] = dn
	return D
#**************************************************************************************************
#**************************************************************************************************
#**************************************************************************************************
#							PARAMETROS INICIALES
path = '/home/sherlock2204f/Kalman_Filter_Series/Dual_Linear_Kalman_Filter_Estimation/'
#path = '/media/sherlock1804/ee29a259-316c-4a85-b9d1-ad38f039a480/home/sherlock2204f/Mis_Documentos/Doctorado/Tesis/Desarrollo_Experimental/servomotor/Dual_Linear_Kalman_Filter_Estimation/'
#file_csv = 'servo.csv'
file_csv = 'sensors_bag_5.csv'

# Parametros del filtro
h = 0.01									# Periodo de muestreo [s]
sigma_Mgamma = 0.009			# Des. Est. de la medicion de la posicion del servomotor
# Parametros del modelo
a1 = 21.0 #3.0 
a2 = 100.0 #8.0 
b = 38.0 #2.0		
# Matrices para el filtro de Kalman para estimar estados
k2 = a2-(a1**2/4.0)
if (k2>0): complex_f = True
else: complex_f = False
F = F_calc(a1,a2,h,complex_f)
Bd = Bd_calc(a1,a2,b,h,complex_f)
Q = Q_calc(sigma_Mgamma,h)
R = np.array([[sigma_Mgamma**2]]) # Si es un escalar, definirlo como matriz de 1x1
H = np.array([[1, 0]])
T,S_gamma,M_gamma = get_data(path,file_csv)		# Vector de las mediciones realizadas
T = T-T[0] # Para medir desde el segundo cero
#*********************************INICIO DEL CICLO*********************************
# Inicializacion
X0 = np.array([0.0,0.0])		# Estado inicial
X0 = np.reshape(X0,(2,1))
P0 = 1.0*np.diag((1.0,10.0))

H0 = np.array([[a2, a1, b]])
h0 = np.array([[a2],[a1],[b]])
Ph0 = 2.0*np.diag((1.0,1.0,1.0))


Xk = []
hk = []
DDM_gamma = []
d0 = 0.0
lambda0 = 1.0 #1.1
eps = 0.75 #0.75
i = 0
N,_ = X0.shape 
_,m = S_gamma.shape
_,l = M_gamma.shape
for y in M_gamma:
	# Ajuste de tamanos
	yn = np.reshape(y,(l,1))
	un = np.reshape(S_gamma[i,:],(m,1))
	# Extrapolacion de estados
	Xn1n = Kalman.states_ext(F,Bd,X0,un)
	Pn1n_x = Kalman.X_covariance_ext(F,P0,Q)
	# Extrapolacion de parametros
	hn1n = Kalman.parameters_ext(h0)
	Pn1n_p = Kalman.P_covariance_ext(Ph0,lambda0)
	# Actualizacion de estados
	Knn_x = Kalman.X_kalmanG_act(Pn1n_x,H,R)
	Xnn = Kalman.states_act(Knn_x,H,Xn1n,yn)
	Pnn_x = Kalman.X_covarianze_act(Pn1n_x,H,Knn_x,R,N)

	# Actualizacion de parametros
	#dn = np.reshape(d,(l,1))
	#xn = np.reshape(Xe[i,:],(n,1))
	xe = np.concatenate((-Xnn,un),axis=0)
	dn = (Xnn[1,0]-d0)/h
	Knn_p = Kalman.P_kalmanG_act(Pn1n_p,H0,eps)
	hnn = Kalman.parameters_act(hn1n,Knn_p,H0,dn,xe)
	Pnn_p = Kalman.P_covarianze_act(Pn1n_p,H0,Knn_p)
	
	a1f = hnn[1][0]
	a2f = hnn[0][0]
	bf = hnn[2][0]
	k2 = a2f-(a1f**2/4.0)
	if (k2>0): complex_f = True
	else: complex_f = False
	F = F_calc(a1f,a2f,h,complex_f)
	Bd = Bd_calc(a1f,a2f,bf,h,complex_f)
		
	# Realimentacion del algoritmo
	X0 = Xnn
	P0 = Pnn_x
	H0 = np.transpose(hnn)
	h0 = hnn
	Ph0 = Pnn_p
	d0 = Xnn[1,0]
	i = i+1
	Xk.append(Xnn)
	hk.append(hnn)
	DDM_gamma.append(dn)			

Xk = np.array(Xk)
hk = np.array(hk)
DDM_gamma = np.array(DDM_gamma)
print('a2 a1 b')
print(hk[-1,:])



"""
# Guardado de datos
states = np.concatenate((T,S_gamma),axis=1)
states = np.concatenate((states,Xk[:,0]),axis=1)
states = np.concatenate((states,Xk[:,1]),axis=1)
np.save('states_ex.npy', states)
"""
a1f = hk[-1,1][0]
a2f = hk[-1,0][0]
bf = hk[-1,2][0]
Par_q0 = np.array([a1f,a2f,bf,0.0,0.0])
np.save('Par_q0.npy', Par_q0)

"""
# Coparacion con la derivada numerica y el filtro pasa-altos
Dn = []
Df = []
N,_ = T.shape
fc = 47.7465
wc = 2*np.pi*fc
df0 = 0.0
for i in range (0,N):
	# Filtro pasa-altos
	df = (1.0/(1.0+h*wc))*(wc*(M_gamma[i]-M_gamma[i-1])+df0) 
	# Derivada numerica
	dn = (M_gamma[i]-M_gamma[i-1])/h 
	df0 = df
	Dn.append(dn)
	Df.append(df)
Dn = np.array(Dn)
Df = np.array(Df)
"""

# Graficas
plot1 = plt.figure(1)
plt.plot(T,M_gamma,'r-',T,Xk[:,0],'bo')
plt.xlabel('[s]')
plt.ylabel('M_gamma')
plt.legend(['mediciones','actualizaciones'])
plt.grid()

plot2 = plt.figure(2)
plt.plot(T,Xk[:,0],'r',T,Xk[:,1],'b',T,DDM_gamma,'g')
plt.xlabel('[s]')
plt.legend(['M_gamma','DM_gamma','DDM_gamma'])
plt.grid()
"""
plot3 = plt.figure(3)
plt.plot(T[1:],Xk[1:,1],'b',T,Dn,'r',T,Df,'y')
plt.xlabel('[s]')
plt.legend(['DM_gamma','Dnum_gamma','Df_gamma'])
plt.grid()
"""
plot4 = plt.figure(4)
plt.plot(T,hk[:,0],'b',T,hk[:,1],'g',T,hk[:,2],'r')
plt.xlabel('[s]')
plt.legend(['a2','a1','b'])
plt.grid()

plt.show()
