import csv
import numpy as np
import matplotlib.pyplot as plt
from lib_kalman import linear_kalman_filter

Kalman = linear_kalman_filter()

#**************************************************************************************************
#**************************************************************************************************
#**************************************************************************************************
def get_data(path,file_csv):
	X = []
	Y = []
	with open(path+file_csv, 'r') as datafile:
		ploting = csv.reader(datafile, delimiter=',')
		for ROWS in ploting:
			X.append(float(ROWS[0]))
			Y.append(float(ROWS[1]))
	return np.array([X,Y]).T
#**************************************************************************************************
#**************************************************************************************************
#**************************************************************************************************

#****************************************************************PARAMETROS INICIALES
# Mediciones de la posición del auto
path = '/home/sherlock2204f/Kalman_Filter/Linear_Kalman_Filter/'
file_csv = 'auto.csv'
Z = get_data(path,file_csv)		# Vector de las mediciones realizadas
h = 1.0 											# Periodo de muestreo [s]
sigma_x = 3.0 								# Des. Est. de la medición de la posición [m]
sigma_y = sigma_x
sigma_a = 0.2									# Desv. Est. de la aceleracion [m/s**2]

# Matriz de transicion de estados
F = np.array([[1,h,0.5*h**2,0,0,0],[0,1,h,0,0,0],[0,0,1,0,0,0],[0,0,0,1,h,0.5*h**2],[0,0,0,0,1,h],[0,0,0,0,0,1]])
# Matrix de ruido
Q = (sigma_a**2)*np.array([[0.25*h**4,0.5*h**3,0.5*h**2,0,0,0],[0.5*h**3,h**2,h,0,0,0],[0.5*h**2,h,1,0,0,0],[0,0,0,0.25*h**4,0.5*h**3,0.5*h**2],[0,0,0,0.5*h**3,h**2,h],[0,0,0,0.5*h**2,h,1]])
# Insertidumbre de las mediciones
R = np.array([[sigma_x**2,0],[0,sigma_y**2]])
# Matriz de observacion
H = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]])
# Matriz de control
B = np.zeros((6,1))
# Senal de control
N,_ = Z.shape
un = np.zeros((N,1))


#****************************************************************INICIO DEL CICLO	
# Inicializacion
X0 = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])		# Estado inicial
P0 = np.diag((500.0,500.0,500.0,500.0,500.0,500.0))

#Kalman.P_flag = False
Xk = Kalman.kalman_offline(X0,P0,F,B,un,H,R,Q,Z)


plt.plot(Z[:,0],Z[:,1],'r-',Xk[:,0],Xk[:,3],'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['mediciones','actualizaciones'])
plt.grid()
plt.axis('scaled')
plt.show()
