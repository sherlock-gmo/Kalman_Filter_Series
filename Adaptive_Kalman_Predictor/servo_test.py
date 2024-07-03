import csv
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Modelo del servomotor
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
#States = np.load('states_ex.npy')
#States = np.load('states_duffin.npy')
States = np.load('states_bag_275.npy')
T = States[:,0]
S_gamma = States[:,1]
M_gamma = States[:,2]
DM_gamma = States[:,3]
tf = T[-1]

# Parametros del predictor
h = 0.01
N = T.shape
N = N[0]
E = np.zeros((N,1))
M_pred = np.zeros((N,1))
DM_pred = np.zeros((N,1))

Par_q0 = np.load('Par_q0.npy')
a1 = Par_q0[0]
a2 = Par_q0[1]
b = Par_q0[2]
c = Par_q0[3]
d = Par_q0[4]
print('a1 a2 b c d')
print(Par_q0)

"""
a1 = 11.8 
a2 = 107.7
b = 41.2 
c = 1.2 
d = -2.6 
"""
"""
a1 = 20.0 
a2 = 100.0 
b = 38.0 
c = 1.0 
d = -1.0 
"""

P = [a1,a2,b,c,d]
M10 = M_pred[0]
M20 = DM_pred[0]
for i in range(1,N):
	M2, M1 = runge_kutta04_2ord(P, S_gamma[i], M20, M10, h)
	M20 = M2
	M10 = M1
	M_pred[i] = M1 
	DM_pred[i] = M2
	E[i] = E[i-1]+(M_gamma[i]-M_pred[i])**2

print('IEC ',E[-1])

plot1 = plt.figure(1)
plt.plot(T,0.4*S_gamma,'k',T,M_gamma,'r',T,M_pred,'b')
plt.xlabel('t [s]')
plt.legend(["S_gamma_scaled","M_gamma","M_pred"])
plt.xlim([0,tf])
plt.ylim([-0.25,0.375])
plt.grid()

plot2 = plt.figure(2)
plt.plot(T,DM_gamma,'r',T,DM_pred,'g')
plt.xlabel('t [s]')
plt.legend(["DM_gamma","DM_pred"])
plt.grid()

plot3 = plt.figure(3)
plt.plot(T,E,'b')
plt.xlabel('t [s]')
plt.legend(["IEC"])
plt.grid()

plt.show()
