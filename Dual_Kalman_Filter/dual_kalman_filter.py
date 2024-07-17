import csv
import numpy as np
import matplotlib.pyplot as plt
from lib_dual_kalman import dual_kalman_filter
from lib_model import model_disc

DKalman = dual_kalman_filter()
Model = model_disc()

#**************************************************************************************************
#**************************************************************************************************
#**************************************************************************************************
# OBTENCION DE LOS DATOS
path = '/home/sherlock2204f/Kalman_Filter_Series/Dual_Kalman_Filter/'
file_csv = 'duffin.csv'
T,theta_ref,theta = Model.get_data(path,file_csv)		# Mediciones obtenidas del experimento
T = T-T[0]				# Para medir desde el segundo cero
tf = T[-1]
Nd,m = theta_ref.shape
_,l = theta.shape

# PARAMETROS INICIALES DEL MODELO
a1 = 20.001	
a2 = 100.0
b = 38.0
c = 0.0
d = 0.0

# MATRICES PARA EL FILTRO DE KALMAN
h = 0.01									# Periodo de muestreo [s]
sigma_Mgamma = 0.009			# Des. Est. de la medicion de la posicion del servomotor
k2 = a2-(a1**2/4.0)				# Verifica si los polos son reales o complejos
if (k2>0): complex_f = True
else: complex_f = False
# Si hay escalares, definirlos como matriz de 1x1
F = Model.F_calc(a1,a2,h,complex_f)				# Matriz de transicion de estados
Bd = Model.BWd_calc(a1,a2,b,h,complex_f)	# Matriz de control discreta
#Wd = Model.BWd_calc(a1,a2,w,h,complex_f)	# Matriz de ruido del proceso discreta
C = np.array([[1.0, 0.0]])								# Matriz de observacion
Q = Model.Q_calc(sigma_Mgamma,h)					# Matriz de ruido de medicion
R = np.array([[sigma_Mgamma**2]]) 				# Matriz R

param = [a1, a2, b, c, d]	

#*********************************ALGORITMO DE KALMAN*********************************
# Inicializacion de los estados
i = 0
Xk = []
X0 = np.array([-0.009744,0.0])		# Estado inicial
X0 = np.reshape(X0,(2,1))
P0 = 1.0*np.diag((1.0,1.0))				# Matriz de covarianza inicial
N,_ = X0.shape 

# Inicializacion de los parametros
H0 = np.array([[a2, a1, b]])
sigma0 = np.array([[a2],[a1],[b]])
Psigma0 = 2.0*np.diag((1.0,1.0,1.0))


sigmak = []
DDtheta = []
delta0 = 0.0
lambda0 = 1.24 #1.1
eps = 0.85 #0.75
i = 0


for th in theta:
	yn = np.reshape(th,(l,1))											# Medicion
	un = np.reshape(theta_ref[i,:],(m,1))					# Entrada de control

	# Extrapolacion de estados
	Xn1n = DKalman.states_ext(F,Bd,X0,un)
	Pn1n_x = DKalman.X_covariance_ext(F,P0,Q)

	# Extrapolacion de parametros
	sigman1n = DKalman.parameters_ext(sigma0)
	Pn1n_p = DKalman.P_covariance_ext(Psigma0,lambda0)
	
	# Actualizacion de los estados
	Knn_x = DKalman.X_kalmanG_act(Pn1n_x,C,R)
	Xnn = DKalman.states_act(Knn_x,C,Xn1n,yn)
	Pnn_x = DKalman.X_covarianze_act(Pn1n_x,C,Knn_x,R,N)
	#Xk.append(Xnn)																# Estados del FK [theta, Dtheta]

	# Actualizacion de parametros
	xe = np.concatenate((-Xnn,un),axis=0)
	deltan = (Xnn[1,0]-delta0)/h
	Knn_p = DKalman.P_kalmanG_act(Pn1n_p,H0,eps)
	sigmann = DKalman.parameters_act(sigman1n,Knn_p,H0,deltan,xe)
	Pnn_p = DKalman.P_covarianze_act(Pn1n_p,H0,Knn_p)

	a1f = sigmann[1][0]
	a2f = sigmann[0][0]
	bf = sigmann[2][0]
	k2 = a2f-(a1f**2/4.0)
	if (k2>0): complex_f = True
	else: complex_f = False
	F = Model.F_calc(a1f,a2f,h,complex_f)
	Bd = Model.BWd_calc(a1f,a2f,bf,h,complex_f)

	# Realimentacion del algoritmo
	X0 = Xnn
	P0 = Pnn_x
	#param = [a1, a2, b, c, d]
	i = i+1


	H0 = np.transpose(sigmann)
	sigma0 = sigmann
	Psigma0 = Pnn_p
	delta0 = Xnn[1,0]

	Xk.append(Xnn)
	sigmak.append(sigmann)
	DDtheta.append(deltan)	

	
Xk = np.array(Xk)
sigmak = np.array(sigmak)
print('a1 a2 b c d ')
print(sigmak[-1,:,0])

#	Guardado de datos
#states = np.concatenate((T,theta_ref),axis=1)
#states = np.concatenate((states,Xk[:,0]),axis=1)
#states = np.concatenate((states,Xk[:,1]),axis=1)
#np.save('states.npy', states)
np.save('param.npy', np.array(param))

# GRAFICAS
plot1 = plt.figure(1)
plt.plot(T,theta_ref,'k',T,theta,'r.',T,Xk[:,0],'b',linewidth=3.0)
plt.xlabel('[s]')
plt.legend(['theta_ref','theta','theta__dual_kalman'],loc='upper right',prop={'size': 16})
plt.xlim([0,55]) #tf
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

plot3 = plt.figure(3)
plt.plot(T,sigmak[:,0,0],'r',T,sigmak[:,1,0],'g',T,sigmak[:,2,0],linewidth=2.5)
plt.xlabel('t [s]')
plt.legend(["a1_hat","a2_hat","b_hat"],loc='upper right',prop={'size': 18})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([0,tf]) #tf
plt.grid()



plt.show()
