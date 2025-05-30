import numpy as np
from src.classes import Lorentz3
import matplotlib.pyplot as plt
import seaborn as sns
ic_seed=0
sigma_o = 1.0
sigma_b= 1.0
sigma_n=3.0
T=2000
discard_len=1000
sigma = 10.; beta = 8/3; rho = 28.
model = Lorentz3(sigma = sigma, beta = beta, rho = rho,ic_seed=0,dt=0.01,int_steps=10,obs_noise=sigma_o,noise_seed=0)
x_o,x_t = np.ascontiguousarray(model.run(T+1,discard_len))
#define observational operator H here, such that y_o=H x_o
H_obs=np.array([[1,0,0],[0,1,0]])
#H_obs=np.eye(3)
y_o = model.observe(H_obs,x_o) #
P_b=sigma_b**2 * np.eye(3)
#P_b[0,0]=0.44
P_b[0,0]=0.47
P_b[1,1]=1.10
R_o=sigma_o**2 * np.eye(H_obs.shape[0])
x_a,x_b,d_ob,d_ab,d_oa=model.optimal_interpolation(H_obs,y_o,P_b,R_o,x_t,sigma_n,T)
print(np.mean((x_a-x_t[1:])**2))
print(np.mean((x_b-x_t[1:])**2))
print(d_ob.T @ d_ob/T)
print(d_ab.T @ d_ob/T)
print(d_oa.T @ d_ob/T)
print(d_ab.T @ d_oa/T)
print(np.trace((d_ab.T@d_oa)/T)/3)
trace_Pb=np.zeros(T)
for i in range(T):
    diff=(x_a[i]-x_t[i]).reshape(1,-1)
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
axs[2].set_xlabel(f'Iteration (dt = {model.dt*model.int_steps})')
fig.suptitle('x_o with sigma_noise='+str(sigma_o))
plt.rcParams.update({'font.size': 22})
fig2, axs2 = plt.subplots(H_obs.shape[0], 1, sharex = True,figsize=(10,10))
for i in range(H_obs.shape[0]):
    axs2[i].plot(y_o[:1000,i], 'r', label = 'y_o')
    axs2[i].plot((x_t@H_obs.T)[:1000,i],'k',label = 'H_obs @ x_t')
    axs2[i].set_ylabel('y'+str(i+1)+'(t)')
axs2[0].legend()
axs2[-1].set_xlabel(f'Iteration (dt = {model.dt*model.int_steps})')
np.set_printoptions(precision=2, suppress=True)
str2=str(np.matrix(H_obs))
fig2.suptitle(" ".join(['y_o with H =', str2])+"sigma_o="+str(sigma_o))
plt.rcParams.update({'font.size': 22})

fig3, axs = plt.subplots(3, 1, sharex = True,figsize=(10,10))
axs[0].plot(x_b[:1000,0], 'g', label = 'x_b')
axs[1].plot(x_b[:1000,1], 'g', label = 'x_b')
axs[2].plot(x_b[:1000,2], 'g', label = 'x_b')
axs[0].plot(x_t[:1000,0], 'k', label = 'x_t')
axs[1].plot(x_t[:1000,1], 'k', label = 'x_t')
axs[2].plot(x_t[:1000,2], 'k', label = 'x_t')
axs[0].set_ylabel('x1(t)')
axs[1].set_ylabel('x2(t)')
axs[2].set_ylabel('x3(t)')
axs[0].legend()
axs[2].set_xlabel(f'Iteration (dt = {model.dt*model.int_steps})')
fig3.suptitle('x_b with no DA sigma_n='+str(sigma_n))

fig4, axs = plt.subplots(3, 1, sharex = True,figsize=(10,10))
axs[0].plot(x_a[:1000,0], 'g', label = 'x_a')
axs[1].plot(x_a[:1000,1], 'g', label = 'x_a')
axs[2].plot(x_a[:1000,2], 'g', label = 'x_a')
axs[0].plot(x_t[:1000,0], 'k', label = 'x_t')
axs[1].plot(x_t[:1000,1], 'k', label = 'x_t')
axs[2].plot(x_t[:1000,2], 'k', label = 'x_t')
axs[0].set_ylabel('x1(t)')
axs[1].set_ylabel('x2(t)')
axs[2].set_ylabel('x3(t)')
axs[0].legend()
axs[2].set_xlabel(f'Iteration (dt = {model.dt*model.int_steps})')
fig4.suptitle('assimilation with y_o with sigma_noise='+str(sigma_o))

fig5, axs5 = plt.subplots(H_obs.shape[0], 1, sharex = True,figsize=(5,10))
for i in range(H_obs.shape[0]):
    axs5[i].plot(d_ob[:1000,i], 'r', label = 'd_ob')
    axs5[i].set_ylabel('y'+str(i+1)+'(t)')
axs2[0].legend()
axs2[-1].set_xlabel(f'Iteration (dt = {model.dt*model.int_steps})')
np.set_printoptions(precision=2, suppress=True)
str5=str(np.matrix(H_obs))
fig5.suptitle(" ".join(['d_ob with H =', str5]))
fig6, axs6 = plt.subplots(1, 1, sharex = True,figsize=(10,10))
axs6.plot(trace_Pb)
fig6.suptitle('trace of Pb vs time')
fig7, axs7 = plt.subplots(4,2, sharex = True,figsize=(10,10))
axs7=axs7.flatten()
sns.heatmap(d_ob.T @ d_ob/T,ax=axs7[0],vmin=0,vmax=2.2)
axs7[0].set_title("E(d_ob d_ob^T)")
sns.heatmap(H_obs@P_b@H_obs.T+R_o,ax=axs7[1],vmin=0,vmax=2.2)
axs7[1].set_title("HP_bH^T + R_o")
sns.heatmap(d_ab.T @ d_ob/T,ax=axs7[2],vmin=0,vmax=2.2)
axs7[2].set_title("E(d_ab d_ob^T)")
sns.heatmap(H_obs@P_b@H_obs.T,ax=axs7[3],vmin=0,vmax=2.2)
axs7[3].set_title("HP_bH^T")
sns.heatmap(d_oa.T @ d_ob/T,ax=axs7[4],vmin=0,vmax=2.2)
axs7[4].set_title("E(d_oa d_ob^T)")
sns.heatmap(R_o,ax=axs7[5],vmin=0,vmax=2.2)
axs7[5].set_title("R_o")
sns.heatmap(d_ab.T @ d_oa/T,ax=axs7[6],vmin=0,vmax=2.2)
axs7[6].set_title("E(d_ab d_oa^T)")
sns.heatmap(H_obs@((x_a-x_t[1:]).T @ (x_a-x_t[1:])/T)@H_obs.T,ax=axs7[7],vmin=0,vmax=2.2)
axs7[7].set_title("HP^aH^T")
fig7.suptitle('Correlation matrices comparision with theory with T='+str(T))
fig7.tight_layout()
plt.show()