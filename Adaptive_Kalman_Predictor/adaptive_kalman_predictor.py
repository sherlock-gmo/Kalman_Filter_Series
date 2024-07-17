import csv
import numpy as np
import matplotlib.pyplot as plt
from lib_kalman_predictor import linear_kalman_filter
from lib_model import model_disc

Kalman = linear_kalman_filter()
Model = model_disc()

#**************************************************************************************************
#**************************************************************************************************
#**************************************************************************************************
# OBTENCION DE LOS DATOS
path = '/home/sherlock2204f/Kalman_Filter_Series/Adaptive_Kalman_Predictor/'
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
w = 0.0

# MATRICES PARA EL FILTRO DE KALMAN
h = 0.01									# Periodo de muestreo [s]
sigma_Mgamma = 0.009			# Des. Est. de la medicion de la posicion del servomotor
k2 = a2-(a1**2/4.0)				# Verifica si los polos son reales o complejos
if (k2>0): complex_f = True
else: complex_f = False
# Si hay escalares, definirlos como matriz de 1x1
F = Model.F_calc(a1,a2,h,complex_f)				# Matriz de transicion de estados
Bd = Model.BWd_calc(a1,a2,b,h,complex_f)	# Matriz de control discreta
Wd = Model.BWd_calc(a1,a2,w,h,complex_f)	# Matriz de ruido del proceso discreta
C = np.array([[1.0, 0.0]])								# Matriz de observacion
Q = Model.Q_calc(sigma_Mgamma,h)					# Matriz de ruido de medicion
R = np.array([[sigma_Mgamma**2]]) 				# Matriz R

#	MATRICES PARA EL PREDICTOR
GAMMA = np.array([20.0,10.0,150.0,10.0,0.25]) # gamma1, gamma2,...,gamma5
ALPHA = [100.0, 20.0] 										# alpha1, alpha2 
param_hat = np.zeros((Nd,5))							# Parametros estimados por el predictor
th_hat = np.zeros((Nd,1))									# Primer estado estimado por el predictor
Dth_hat = np.zeros((Nd,1))								# Segundo estado estimado por el predictor
E = np.zeros((Nd,1))											# Guardado de la IEC
param = [a1, a2, b, c, d]	

#*********************************ALGORITMO DE KALMAN*********************************
# Inicializacion de los estados
i = 0
Xk = []
X0 = np.array([-0.009744,0.0])		# Estado inicial
X0 = np.reshape(X0,(2,1))
P0 = 1.0*np.diag((1.0,1.0))				# Matriz de covarianza inicial
N,_ = X0.shape 

for th in theta:
	yn = np.reshape(th,(l,1))											# Medicion
	un = np.reshape(theta_ref[i,:],(m,1))					# Entrada de control
	# Extrapolacion
	Xn1n = Kalman.states_ext(F,Bd,Wd,X0,un)				# Extrapolacion de los estados
	Pn1n = Kalman.covariance_ext(F,P0,Q)					# Extrapolacion de la matriz de covarianza
	# Actualizacion de los estados
	Knn = Kalman.kalmanG_act(Pn1n,C,R)						# Calculo de la ganancia de Kalman
	Xnn = Kalman.states_act(Knn,C,Xn1n,yn)				# Actualizacion de los estados
	Pnn = Kalman.covarianze_act(Pn1n,C,Knn,R,N)		# Actualizacion de la matriz de covarianza
	Xk.append(Xnn)																# Estados del FK [theta, Dtheta]
	# Predictor
	V = [theta_ref[i],Xk[i][0],Xk[i][1]]
	a1, a2, b, c, d, Dth_h, th_h = Kalman.predictor(param,V,ALPHA,GAMMA,Dth_hat[i-1],th_hat[i-1],h)
	# Actualizacion de las matrices del modelo
	w = d-c*Xk[i][1][0]/abs(Xk[i][1][0])
	k2 = a2-(a1**2/4.0)														# Verifica si los polos son reales o complejos
	if (k2>0): complex_f = True
	else: complex_f = False
	F = Model.F_calc(a1,a2,h,complex_f)						# Actualizacion de la matriz de transicion de estados
	Bd = Model.BWd_calc(a1,a2,b,h,complex_f)			# Actualizacion de la matriz de control discreta
	Wd = Model.BWd_calc(a1,a2,w,h,complex_f)			# Actualizacion de la matriz de ruido del proceso discreta
	# Guardado de datos
	param_hat[i,:] = [a1,a2,b,c,d]									# Parametros estimados por el predictor
	th_hat[i] = th_h																# Primer estado estimado por el predictor
	Dth_hat[i] = Dth_h															# Segundo estado estimado por el predictor
	E[i] = E[i-1]+(Xk[i][0]-th_h)**2								# Integral de error cuadratico del primer estado
	# Realimentacion del algoritmo
	X0 = Xnn
	P0 = Pnn
	param = [a1, a2, b, c, d]
	i = i+1
	
Xk = np.array(Xk)
print('a1 a2 b c d ')
print(param_hat[-1,:])

#	Guardado de datos
#states = np.concatenate((T,theta_ref),axis=1)
#states = np.concatenate((states,Xk[:,0]),axis=1)
#states = np.concatenate((states,Xk[:,1]),axis=1)
#np.save('states.npy', states)
np.save('param.npy', np.array(param))

# GRAFICAS
plot1 = plt.figure(1)
plt.plot(T,theta_ref,'k',T,theta,'r.',T,Xk[:,0],'b',T,th_hat,'g',linewidth=3.0)
plt.xlabel('[s]')
plt.legend(['theta_ref','theta','theta_kalman','theta_pred'],loc='upper right',prop={'size': 16})
plt.xlim([0,55]) #tf
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

plot2 = plt.figure(2)
plt.plot(T,E,'b',linewidth=2.5)
plt.xlabel('t [s]')
plt.legend(["IEC"],loc='lower right',prop={'size': 18})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True)
AX = [ax1, ax2, ax3, ax4, ax5]
colors = ['r','g','m','b','y']
names = ["a1_hat","a2_hat","b_hat","c_hat","d_hat"]
i = 0
for ax in AX:
	ax.plot(T,param_hat[:,i],colors[i],label=names[i])
	ax.legend(loc="lower left",prop={'size': 18})
	ax.tick_params(axis='x', labelsize=20)
	ax.tick_params(axis='y', labelsize=20)
	ax.set_xlim([0,tf])
	ax.grid()
	i=i+1

plt.show()
