import numpy as np
from src.classes import Lorentz3
import matplotlib.pyplot as plt
import seaborn as sns
ic_seed=0
sigma_o = 1.0
sigma_b= 0.6
sigma_q= 0.6
sigma_n=3.0
T=2000
discard_len=1000
sigma = 10.; beta = 8/3; rho = 28.
model = Lorentz3(sigma = sigma, beta = beta, rho = rho,ic_seed=0,dt=0.01,int_steps=10,obs_noise=sigma_o,noise_seed=0)
x_o,x_t = np.ascontiguousarray(model.run(T+1,discard_len))
#define observational operator H here, such that y_o=H x_o
H_obs=np.array([[1,0,0],[0,1,0]])
H_obs=np.eye(3)
y_o = model.observe(H_obs,x_o)
P_b=sigma_b**2 * np.eye(3)
Q_b=sigma_q**2 * np.eye(3)
#for full state
Q_b[0,0]=0.14
P_b[0,0]=0.14
#for no z
#Q_b[0,0]=0.23
#P_b[0,0]=0.23
#Q_b[1,1]=0.5
#P_b[1,1]=0.5
R_o=sigma_o**2 * np.eye(H_obs.shape[0])
x_a,x_b,d_ob,d_ab,d_oa,P_bs,P_as=model.EKF(H_obs,y_o,P_b,R_o,Q_b,x_t,sigma_n,T)
eigv_bs=np.zeros((T+1))
sym_ness_b=np.zeros((T+1))
eigv_as=np.zeros((T+1))
sym_ness_a=np.zeros((T+1))
# check positive definiteness and symmetry of P_a and P_b
for i in range(T+1):
    eigv_b,_=np.linalg.eigh(P_bs[i])
    eigv_a, _ = np.linalg.eigh(P_as[i])
    eigv_bs[i]=eigv_b[0]
    eigv_as[i] = eigv_a[0]
    sym_ness_b[i]=np.mean((P_bs[i]-(P_bs[i]+P_bs[i].T)/2)**2)
    sym_ness_a[i] = np.mean((P_as[i]-(P_as[i]+P_as[i].T)/2)**2)
# check ratio_boa<1, cos_boa>0
d_oa_norm=np.sqrt(np.sum(d_oa**2,axis=1))
d_ob_norm=np.sqrt(np.sum(d_ob**2,axis=1))
cos_boa=np.diag(d_ob@d_oa.T)/(d_oa_norm*d_ob_norm)
ratio_boa=d_oa_norm/d_ob_norm
print(np.mean((x_a-x_t[1:])**2))
print(np.mean((x_b-x_t[1:])**2))
#verify TLM
TLM_verify_time=200
num_initial_conditions=1000
#data varify
data_v=x_t[:num_initial_conditions]
delta_x_TLM,delta_x_non_linear=model.var_TLM(data_v,T=TLM_verify_time,sigma_p=1.0)
MSE_delta_X_TLM_non_linear=np.mean((delta_x_TLM[1:,:,:]-delta_x_non_linear[1:,:,:])**2,axis=1)
print(np.amax(MSE_delta_X_TLM_non_linear))
delta_X_non_linear_norm=np.sqrt(np.sum(delta_x_non_linear**2,axis=1))
delta_X_TLM_norm=np.sqrt(np.sum(delta_x_TLM**2,axis=1))
cos_TLM_non_linear=np.zeros((TLM_verify_time+1,data_v.shape[0]))
for i in range(data_v.shape[0]):
    cos_TLM_non_linear[:,i]=(np.diag(delta_x_TLM[:,:,i] @ delta_x_non_linear[:,:,i].T)/(delta_X_non_linear_norm[:,i]*delta_X_TLM_norm[:,i])).reshape(TLM_verify_time+1,)
trace_Pb=np.zeros(T)
for i in range(T):
    diff=(x_b[i]-x_a[i]).reshape(1,-1)
    trace_Pb[i]= np.trace(diff.T @ diff)
plt.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(3, 1, sharex = True,figsize=(10,10))
axs[0].plot(x_o[:1000,0], 'r', label = 'x_o')
axs[1].plot(x_o[:1000,1], 'r', label = 'x_o')
axs[2].plot(x_o[:1000,2], 'r', label = 'x_o')
axs[0].set_ylabel('x1(t)')
axs[1].set_ylabel('x2(t)')
axs[2].set_ylabel('x3(t)')
axs[0].legend()
axs[2].set_xlabel(f'Iteration (dt = {model.dt})')
fig.suptitle('x_o with sigma_noise='+str(sigma_o))
plt.rcParams.update({'font.size': 22})
fig2, axs2 = plt.subplots(H_obs.shape[0], 1, sharex = True,figsize=(10,10))
for i in range(H_obs.shape[0]):
    axs2[i].plot(y_o[:1000,i], 'r', label = 'y_o')
    axs2[i].plot((x_t@H_obs.T)[:1000,i],'k',label = 'H_obs @ x_t')
    axs2[i].set_ylabel('y'+str(i+1)+'(t)')
axs2[0].legend()
axs2[-1].set_xlabel(f'Iteration (dt = {model.dt})')
np.set_printoptions(precision=2, suppress=True)
str2=str(np.matrix(H_obs))
fig2.suptitle(" ".join(['y_o with H =', str2]))
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
axs[2].set_xlabel(f'Iteration (dt = {model.dt})')
fig3.suptitle('x_b with no DA and sigma_n ='+str(sigma_n))

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
fig8, axs8 = plt.subplots(1, 1, sharex = True,figsize=(10,10))
axs8.scatter(eigv_as,eigv_bs)
axs8.set_xlabel("smallest eigenvalue of P_a")
axs8.set_ylabel("smallest eigenvalue of P_b")
axs8.set_title("smallest eigenvalue of P_a and P_b of T="+str(T))
fig9, axs9 = plt.subplots(2, 1, sharex = True,figsize=(10,10))
axs9=axs9.flatten()
axs9[0].plot(sym_ness_b)
axs9[1].plot(sym_ness_a)
axs9[0].set_title("MSE of P_b with (P_b+P_b^T)/2")
axs9[1].set_title("MSE of P_a with (P_a+P_a^T)/2")
fig10, axs10 = plt.subplots(2, 1, sharex = True,figsize=(10,10))
axs10=axs10.flatten()
axs10[0].plot(cos_boa)
axs10[1].plot(ratio_boa)
axs10[1].axhline(y=1, color='r', linestyle='-')
axs10[0].set_title("cos theta_boa")
axs10[1].set_title("|d_oa|/|d_ob|")
fig11, axs11 = plt.subplots(2, 1, sharex = True,figsize=(10,10))
axs11=axs11.flatten()
sns.heatmap(np.log(MSE_delta_X_TLM_non_linear+10**(-35))/np.log(10),ax=axs11[0])
sns.heatmap(cos_TLM_non_linear,ax=axs11[1],vmin=-1,vmax=1)
axs11[0].set_title("MSE between delta_X_TLM and delta_non_linear")
axs11[1].set_xlabel("n th initial condition")
axs11[1].set_title("cos theta between delta_X_TLM and delta_non_linear")
fig11.supylabel("# Time Windows (TW="+str(model.int_steps*model.dt)+")")
fig11.tight_layout()
plt.show()