import csv
import numpy as np

class model_disc():
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
	def __init__(self):
		pass
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#		Obtencion de datos
	def get_data(self,path,file_csv):
		T = []
		theta_ref = []
		theta = []
		with open(path+file_csv, 'r') as datafile:
			ploting = csv.reader(datafile, delimiter=',') #\t #,
			for ROWS in ploting:
				T.append(float(ROWS[0]))
				theta_ref.append(float(ROWS[1]))#3 #1
				theta.append(float(ROWS[2]))#4 #2
			N = len(T)
			T = np.reshape(np.array(T).T,(N,1))
			theta_ref = np.reshape(np.array(theta_ref).T,(N,1))
			theta = np.reshape(np.array(theta).T,(N,1))
		return T, theta_ref, theta
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Calculo de la matriz de transicion de estados
	def F_calc(self,a1,a2,h,complex_f):
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
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
	# Calculo de las matrices Bd/Wd
	def BWd_calc(self,a1,a2,bw,h,complex_f):
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
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
	# Calculo de la matriz de ruido de medicion
	def Q_calc(self,sigma_Mgamma,h):
		Q = (sigma_Mgamma**2)*np.array([[1.0, 1.0/h],[1.0/h, 1.0/(h**2)]])
		return Q

