import numpy as np
from src.classes import Lorentz3
import matplotlib.pyplot as plt
ic_seed=0
sigma_o = 1
T=1000
discard_len=1000
sigma = 10.; beta = 8/3; rho = 28.
model = Lorentz3(sigma = sigma, beta = beta, rho = rho,ic_seed=0,dt=0.01,int_steps=1,obs_noise=sigma_o)
x_o = np.ascontiguousarray(model.run(T,discard_len))
#define observational operator H here, such that y_o=H x_o
H_obs=np.array([[1/3,1/3,1/3],[0,1/2,1/2]])
y_o = model.observe(H_obs,x_o) #

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
    axs2[i].plot(y_o[:,i], 'r--', label = 'y_o')
    axs2[i].set_ylabel('y'+str(i+1)+'(t)')
axs2[0].legend()
axs2[-1].set_xlabel(f'Iteration (dt = {model.dt})')
np.set_printoptions(precision=2, suppress=True)
str2=str(np.matrix(H_obs))
fig2.suptitle(" ".join(['y_o with H =', str2]))
plt.show()