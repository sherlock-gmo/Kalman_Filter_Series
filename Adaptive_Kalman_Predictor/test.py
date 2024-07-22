import csv
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
def get_data(path,file_csv):
	T = []
	theta_ref = []
	theta = []
	with open(path+file_csv, 'r') as datafile:
		ploting = csv.reader(datafile, delimiter='\t') #\t #,
		for ROWS in ploting:
			T.append(float(ROWS[0]))
			theta_ref.append(float(ROWS[3]))#3 #1
			theta.append(float(ROWS[4]))#4 #2
		N = len(T)
		T = np.reshape(np.array(T).T,(N,1))
		theta_ref = np.reshape(np.array(theta_ref).T,(N,1))
		theta = np.reshape(np.array(theta).T,(N,1))
	return T, theta_ref, theta
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Modelo del servomotor en forma de ecuacion de estados
def f1(M1,M2):
	return M2
def f2(P,S,M1,M2):
	a1,a2,b,c,d = P
	if (abs(M2)>0.0): sDM = M2/abs(M2)
	else: sDM = 0.0
	DM2 = (-a1)*M2+(-a2)*M1-c*sDM+b*S+d
	return DM2
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Solucion de la ecuacion de estados usando Runge-Kuta
def runge_kutta04_2ord(P, S, M20, M10, h):
	k1 = h * f1(M10, M20)
	l1 = h * f2(P, S, M10, M20)
	k2 = h * f1(M10 + k1/2, M20 + l1/2)
	l2 = h * f2(P, S, M10 + k1/2, M20 + l1/2)
	k3 = h * f1(M10 + k2/2, M20 + l2/2)
	l3 = h * f2(P, S, M10 + k2/2, M20 + l2/2)
	k4 = h * f1(M10 + k3, M20 + l3)
	l4 = h * f2(P, S, M10 + k3, M20 + l3)
	M1 = M10 + (k1 + 2*k2 + 2*k3 + k4) / 6
	M2 = M20 + (l1 + 2*l2 + 2*l3 + l4) / 6
	return M2, M1
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------	
# ESTADOS MEDIDOS POR EL FILTRO DE KALMAN
path = '/home/sherlock2204f/Kalman_Filter_Series/Adaptive_Kalman_Predictor/'
file_csv = 'sensors_bag_225.csv'
T,theta_ref,theta = get_data(path,file_csv)		# Mediciones obtenidas del experimento
T = T-T[0]				# Para medir desde el segundo cero
tf = T[-1]

# ESTADOS OBTENIDOS DEL MODELO MATEMATICO
param = np.load('param.npy')
h = 0.01
N = T.shape[0]
theta_m = np.zeros((N,1))
Dtheta_m = np.zeros((N,1))
IEC = np.zeros((N,1))
# inicializacion de los estados
M10 = theta_m[0]
M20 = Dtheta_m[0]
for i in range(1,N):
	# Solucion de la ecuacion de estados usando Runge-Kuta de 4o orden
	M2, M1 = runge_kutta04_2ord(param, theta_ref[i], M20, M10, h)
	M20 = M2
	M10 = M1
	theta_m[i] = M1 
	Dtheta_m[i] = M2
	IEC[i] = IEC[i-1]+(theta[i]-theta_m[i])**2		# Calculode la IEC

print('a1 a2 b c d')
print(param)
print('IEC ',IEC[-1])
print('RMSE ',np.sqrt(IEC[-1]/N))

# GRAFICAS
plot1 = plt.figure(1)
plt.plot(T,0.38*theta_ref,'k',T,theta,'r',T,theta_m,'b',linewidth=2.5)
plt.xlabel('t [s]')
plt.legend(["theta_ref_N","theta","theta_model"],loc='lower right',prop={'size': 18})
plt.xlim([0,tf])
plt.ylim([-0.25,0.375])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

plot2 = plt.figure(2)
plt.plot(T,IEC,'b',linewidth=2.5)
plt.xlabel('t [s]')
plt.legend(["IEC"],loc='lower right',prop={'size': 18})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

plt.show()
