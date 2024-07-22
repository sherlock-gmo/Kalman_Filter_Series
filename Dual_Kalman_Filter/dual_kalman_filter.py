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
c = 0.001
d = 0.001
w = 0.001
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
H0 = np.array([[a2, a1,b,c,d]])
S0 = H0.T
Psigma0 = 2.0*np.diag((1.0,1.0,1.0,1.0,1.0))

lambda0 = 1.24 #1.1
eps = 0.85 #0.75
Dtheta0 = 0.0
Sk = []
IEC = np.zeros((Nd,1))
i = 0
for th in theta:
	yn = np.reshape(th,(l,1))											# Medicion
	un = np.reshape(theta_ref[i,:],(m,1))					# Entrada de control
	# Extrapolacion de estados
	Xn1n = DKalman.states_ext(F,Bd,Wd,X0,un)
	Pn1n_x = DKalman.X_covariance_ext(F,P0,Q)
	# Extrapolacion de parametros
	Sn1n = DKalman.parameters_ext(S0)
	Pn1n_p = DKalman.P_covariance_ext(Psigma0,lambda0)
	# Actualizacion de los estados
	Knn_x = DKalman.X_kalmanG_act(Pn1n_x,C,R)
	Xnn = DKalman.states_act(Knn_x,C,Xn1n,yn)
	Pnn_x = DKalman.X_covarianze_act(Pn1n_x,C,Knn_x,R,N)
	# Actualizacion de parametros
	if (Xnn[1,0]>0.0): sign_Dth=1.0
	if (Xnn[1,0]==0.0): sign_Dth=0.0
	if (Xnn[1,0]<0.0): sign_Dth=-1.0
	phi = np.array([[-Xnn[0,0]],[-Xnn[1,0]],[un[0][0]],[-sign_Dth],[1.0]])
	delta = (Xnn[1,0]-Dtheta0)/h	
	Knn_p = DKalman.P_kalmanG_act(Pn1n_p,H0,eps)
	Snn = DKalman.parameters_act(Sn1n,Knn_p,H0,delta,phi)
	Pnn_p = DKalman.P_covarianze_act(Pn1n_p,H0,Knn_p)

	a1 = Snn[1][0]
	a2 = Snn[0][0]
	b = Snn[2][0]
	c = Snn[3][0]
	d = Snn[4][0]
	w = d-c*sign_Dth
	k2 = a2-(a1**2/4.0)
	if (k2>0): complex_f = True
	else: complex_f = False
	F = Model.F_calc(a1,a2,h,complex_f)
	Bd = Model.BWd_calc(a1,a2,b,h,complex_f)
	Wd = Model.BWd_calc(a1,a2,w,h,complex_f)	# Matriz de ruido del proceso discreta
	# Realimentacion del algoritmo
	param = [a1, a2, b, c, d]	
	X0 = Xnn
	P0 = Pnn_x
	S0 = Snn
	Psigma0 = Pnn_p
	H0 = np.transpose(Snn)
	Dtheta0 = Xnn[1,0]
	Xk.append(Xnn)
	Sk.append(Snn)
	IEC[i] = IEC[i-1]+(Xk[i][0]-yn)**2			# Integral de error cuadratico del primer estado
	i = i+1
		
Xk = np.array(Xk)
Sk = np.array(Sk)
print('a2 a1 b c d ')
print(Sk[-1,:,0])
print('IEC ',IEC[-1])
print('RMSE ',np.sqrt(IEC[-1]/N))

#	Guardado de datos
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

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True)
AX = [ax1, ax2, ax3, ax4, ax5]
colors = ['r','g','m','b','y']
names = ["a1_hat","a2_hat","b_hat","c_hat","d_hat"]
i = 0
for ax in AX:
	ax.plot(T,Sk[:,i,0],colors[i],label=names[i],linewidth=2.5)
	ax.legend(loc="lower right",prop={'size': 18})
	ax.tick_params(axis='x', labelsize=20)
	ax.tick_params(axis='y', labelsize=20)
	ax.set_xlim([-0.1,5])
	ax.set_ylim([Sk[0,i,0],Sk[-1,i,0]*1.001])
	ax.grid()
	i=i+1

plt.show()
