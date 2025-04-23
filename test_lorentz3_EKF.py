import numpy as np
from src.classes import Lorentz3
import matplotlib.pyplot as plt
ic_seed=0
sigma_o = 1.0
sigma_b= 3.0
T=1000
discard_len=1000
sigma = 10.; beta = 8/3; rho = 28.
model = Lorentz3(sigma = sigma, beta = beta, rho = rho,ic_seed=0,dt=0.01,int_steps=1,obs_noise=sigma_o,noise_seed=0)
x_o,x_t = np.ascontiguousarray(model.run(T+1,discard_len))
#define observational operator H here, such that y_o=H x_o
H_obs=np.eye(3)
y_o = model.observe(H_obs,x_o) #
P_b=sigma_b**2 * np.eye(3)
Q_b=sigma_b**2 * np.eye(3)
R_o=sigma_o**2 * np.eye(H_obs.shape[0])
x_a,x_b,d_ob=model.EKF(H_obs,y_o,P_b,R_o,Q_b,x_t,sigma_b,T)
print(np.mean((x_a-x_t[1:])**2))
print(np.mean((x_b-x_t[1:])**2))
trace_Pb=np.zeros(T)
for i in range(T):
    diff=(x_b[i]-x_a[i]).reshape(1,-1)
    trace_Pb[i]= np.trace(diff.T @ diff)
plt.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(3, 1, sharex = True,figsize=(10,10))
axs[0].plot(x_o[:,0], 'r', label = 'x_o')
axs[1].plot(x_o[:,1], 'r', label = 'x_o')
axs[2].plot(x_o[:,2], 'r', label = 'x_o')
axs[0].set_ylabel('x1(t)')
axs[1].set_ylabel('x2(t)')
axs[2].set_ylabel('x3(t)')
axs[0].legend()
axs[2].set_xlabel(f'Iteration (dt = {model.dt})')
fig.suptitle('x_o with sigma_noise='+str(sigma_o))
plt.rcParams.update({'font.size': 22})
fig2, axs2 = plt.subplots(H_obs.shape[0], 1, sharex = True,figsize=(10,10))
for i in range(H_obs.shape[0]):
    axs2[i].plot(y_o[:,i], 'r', label = 'y_o')
    axs2[i].plot((x_t@H_obs.T)[:,i],'k',label = 'H_obs @ x_t')
    axs2[i].set_ylabel('y'+str(i+1)+'(t)')
axs2[0].legend()
axs2[-1].set_xlabel(f'Iteration (dt = {model.dt})')
np.set_printoptions(precision=2, suppress=True)
str2=str(np.matrix(H_obs))
fig2.suptitle(" ".join(['y_o with H =', str2]))
plt.rcParams.update({'font.size': 22})

fig3, axs = plt.subplots(3, 1, sharex = True,figsize=(10,10))
axs[0].plot(x_b[:,0], 'g', label = 'x_b')
axs[1].plot(x_b[:,1], 'g', label = 'x_b')
axs[2].plot(x_b[:,2], 'g', label = 'x_b')
axs[0].plot(x_t[:,0], 'k', label = 'x_t')
axs[1].plot(x_t[:,1], 'k', label = 'x_t')
axs[2].plot(x_t[:,2], 'k', label = 'x_t')
axs[0].set_ylabel('x1(t)')
axs[1].set_ylabel('x2(t)')
axs[2].set_ylabel('x3(t)')
axs[0].legend()
axs[2].set_xlabel(f'Iteration (dt = {model.dt})')
fig3.suptitle('x_b with sigma_noise='+str(sigma_b))

fig4, axs = plt.subplots(3, 1, sharex = True,figsize=(10,10))
axs[0].plot(x_a[:,0], 'g', label = 'x_a')
axs[1].plot(x_a[:,1], 'g', label = 'x_a')
axs[2].plot(x_a[:,2], 'g', label = 'x_a')
axs[0].plot(x_t[:,0], 'k', label = 'x_t')
axs[1].plot(x_t[:,1], 'k', label = 'x_t')
axs[2].plot(x_t[:,2], 'k', label = 'x_t')
axs[0].set_ylabel('x1(t)')
axs[1].set_ylabel('x2(t)')
axs[2].set_ylabel('x3(t)')
axs[0].legend()
axs[2].set_xlabel(f'Iteration (dt = {model.dt})')
fig4.suptitle('assimilation with y_o with sigma_noise='+str(sigma_o))

fig5, axs5 = plt.subplots(H_obs.shape[0], 1, sharex = True,figsize=(5,10))
for i in range(H_obs.shape[0]):
    axs5[i].plot(d_ob[:,i], 'r', label = 'd_ob')
    axs5[i].set_ylabel('y'+str(i+1)+'(t)')
axs2[0].legend()
axs2[-1].set_xlabel(f'Iteration (dt = {model.dt})')
np.set_printoptions(precision=2, suppress=True)
str5=str(np.matrix(H_obs))
fig5.suptitle(" ".join(['d_ob with H =', str5]))
fig6, axs6 = plt.subplots(1, 1, sharex = True,figsize=(10,10))
axs6.plot(trace_Pb)
fig6.suptitle('trace of Pb vs time')
plt.show()