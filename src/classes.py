from src.model_functions import *
import numpy as np
from numba import njit

class Lorentz3():
    def __init__(self, dt=0.01, int_steps=10, sigma=10.,beta=8 / 3, rho=28., ic=np.array([]), ic_seed=0,obs_noise=1e-5,noise_seed=10):
        self.params = np.array([sigma, rho, beta])
        self.dxdt = lorentz3_dxdt
        self.dxMdt=general_dxMdt
        self.TLM =lorentz3_TLM
        self.dxPdt=general_dxPdt
        self.dt = dt
        self.int_steps = int_steps
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1) * np.array([1., 1., 30.])
        elif ic.size == 3:
            self.state = ic.flatten()
        else:
            raise ValueError
        self.obs_noise=obs_noise
        self.noise_seed=noise_seed

    def run(self, T, discard_len=0): #certain length is discarded for spin up
        output = run_model(self.state, T, self.dxdt, self.params,self.dt, discard_len, self.int_steps)
        self.state = output[-1]
        rng = np.random.default_rng(self.noise_seed)
        return output + self.obs_noise* rng.normal(size=output.shape),output

    def observe(self, H_op, x_o):
        return x_o @ H_op.T
    def optimal_interpolation(self,H_op,y_o,P_b,R_o,x_t,sigma_b,T):
        K_oi = P_b @ H_op.T @ np.linalg.inv(H_op @ P_b @ H_op.T + R_o)
        x_b = np.zeros((T+1,3))
        x_b_non_DA = np.zeros((T + 1, 3))
        x_a = np.zeros((T+1,3))
        d_ob=np.zeros((T+1,y_o.shape[1]))
        d_ab= np.zeros((T + 1, y_o.shape[1]))
        d_oa = np.zeros((T + 1, y_o.shape[1]))
        x_a[0]=x_t[0]+ sigma_b * np.random.normal(size=(3,))
        x_b_non_DA[0] = x_a[0]
        for i in range(T):
            x_b[i+1] = run_step(x_a[i],self.dxdt,self.dt,self.int_steps,self.params)
            x_b_non_DA[i+1]=run_step(x_b_non_DA[i],self.dxdt,self.dt,self.int_steps,self.params)
            d_ob[i+1] = y_o[i+1]-x_b[i+1]@H_op.T
            x_a[i+1] = x_b[i+1]+d_ob[i+1]@K_oi.T
            d_ab[i+1] = x_a[i+1]@H_op.T-x_b[i+1]@H_op.T
            d_oa[i+1] = y_o[i+1]-x_a[i+1]@H_op.T
        return x_a[1:],x_b_non_DA[1:],d_ob[1:],d_ab[1:],d_oa[1:]

    def EKF(self,H_op,y_o,P_b_0,R_o,Qb,x_t,sigma_b,T):
        x_b = np.zeros((T+1,3))
        x_b_non_DA = np.zeros((T + 1, 3))
        x_a = np.zeros((T+1,3))
        d_ob=np.zeros((T+1,y_o.shape[1]))
        d_ab = np.zeros((T + 1, y_o.shape[1]))
        d_oa = np.zeros((T + 1, y_o.shape[1]))
        x_a[0]=x_t[0]+ sigma_b * np.random.normal(size=(3,))
        x_b_non_DA[0] = x_t[0] + sigma_b * np.random.normal(size=(3,))
        P_b=np.zeros((T+1,3,3))
        P_a = np.zeros((T+1, 3, 3))
        P_a[0]=P_b_0
        P_b[0] = P_b_0
        for i in range(T):
            x_b[i+1],P_b[i+1] = run_step_EK(x_a[i],P_a[i],self.dxPdt, self.dxdt,self.TLM,self.dt,self.int_steps,self.params,Qb)
            x_b_non_DA[i+1]=run_step(x_b_non_DA[i],self.dxdt,self.dt,self.int_steps,self.params)
            d_ob[i+1] = y_o[i+1]-x_b[i+1]@H_op.T
            K_oi = P_b[i+1] @ H_op.T @ np.linalg.inv(H_op @ P_b[i+1] @ H_op.T + R_o)
            x_a[i+1] = x_b[i+1]+d_ob[i+1]@K_oi.T
            P_a[i+1] = (np.identity(3)-K_oi @ H_op) @ P_b[i+1]
            d_ab[i + 1] = x_a[i + 1] @ H_op.T - x_b[i + 1] @ H_op.T
            d_oa[i + 1] = y_o[i + 1] - x_a[i + 1] @ H_op.T
        return x_a[1:],x_b_non_DA[1:],d_ob[1:],d_ab[1:],d_oa[1:],P_b,P_a
    def var_TLM(self,data,T=10,seed=1,sigma_p=0.1):
        M=np.zeros((T+1,3,3,data.shape[0]))
        x_TLM=np.zeros((T+1,3,data.shape[0]))
        x_non_linear=np.zeros((T+1,3,data.shape[0]))
        delta_x_non_linear= np.zeros((T+1, 3, data.shape[0]))
        delta_x_TLM = np.zeros((T+1, 3, data.shape[0]))
        rng = np.random.default_rng(seed)
        perturbed_data=data+sigma_p*rng.normal(size=data.shape)
        print(data.shape)
        for j in range(data.shape[0]):
            M[0,:,:,j] = np.array([[1.0,0,0],[0,1.0,0],[0,0,1.0]])
            x_TLM[0,:,j] = data[j]
            x_non_linear[0,:,j] = data[j]
            delta_x_TLM[0,:,j] = perturbed_data[j]-data[j]
            delta_x_non_linear[0, :, j] = perturbed_data[j] - data[j]
            for i in range(T):
                x_TLM[i+1,:,j],M[i+1,:,:,j]=run_step_TLM(np.ascontiguousarray(x_TLM[i,:,j]),np.ascontiguousarray(M[i,:,:,j]),self.dxdt,self.dxMdt,self.TLM,self.dt,self.int_steps,self.params)
                delta_x_TLM[i+1, :, j]=M[i+1,:,:,j]@delta_x_TLM[0,:,j]
                x_non_linear[i+1,:,j],delta_x_non_linear[i+1,:,j]=run_step_non_linear(np.ascontiguousarray(x_non_linear[i, :, j]), np.ascontiguousarray(delta_x_non_linear[i, :, j]), self.dxdt, self.dxMdt, self.TLM, self.dt, self.int_steps,self.params)
        return delta_x_TLM,delta_x_non_linear
    def EnKF(self,H_op,y_o,P_b_0,R_o,x_t,sigma_b,T,rho=1.0,en_size=300):
        x_b = np.zeros((T+1,3))
        x_b_non_DA = np.zeros((T + 1, 3))
        x_a = np.zeros((T+1,3))
        d_ob=np.zeros((T+1,y_o.shape[1]))
        d_ab = np.zeros((T + 1, y_o.shape[1]))
        d_oa = np.zeros((T + 1, y_o.shape[1]))
        x_a[0]=x_t[0]+ sigma_b * np.random.normal(size=(3,))
        x_b[0]=x_a[0]
        x_b_non_DA[0] = x_a[0]
        x_en_b = np.zeros((T+1,en_size,3))
        x_en_a = np.zeros((T + 1, en_size, 3))
        d_ob_en = np.zeros((T + 1,en_size, y_o.shape[1]))
        x_hat_b = np.zeros((T + 1, en_size, 3))
        x_hat_a = np.zeros((T + 1, en_size, 3))
        P_b=np.zeros((T+1,3,3))
        P_a = np.zeros((T + 1, 3, 3))
        P_a[0] = P_b_0
        P_b[0] = P_b_0
        x_en_b[0]=x_a[0]+np.random.multivariate_normal(np.zeros(3), P_b_0,size=en_size)
        x_en_a[0] = x_en_b[0]
        for i in range(T):
            x_b[i+1] = run_step(x_a[i],self.dxdt,self.dt,self.int_steps,self.params)
            x_b_non_DA[i+1]=run_step(x_b_non_DA[i],self.dxdt,self.dt,self.int_steps,self.params)
            d_ob[i+1] = y_o[i+1]-x_b[i+1]@H_op.T
            e_p=np.random.multivariate_normal(np.zeros(y_o.shape[1]),R_o,size=en_size)
            d_ob_en[i+1] = y_o[i+1]+e_p-x_en_b[i+1]@H_op.T
            x_en_b[i+1] = run_array(x_en_a[i],self.dxdt,self.dt,self.int_steps,self.params)
            x_hat_b[i+1] = x_en_b[i+1]-np.mean(x_en_b[i+1],axis=0)
            x_hat_b[i + 1]= rho*x_hat_b[i + 1]
            P_b[i+1] = 1/(en_size-1)*(x_hat_b[i+1].T @ x_hat_b[i+1])
            K_oi = P_b[i+1] @ H_op.T @ np.linalg.inv(H_op @ P_b[i+1] @ H_op.T + R_o)
            x_a[i+1] = x_b[i+1]+d_ob[i+1]@K_oi.T
            x_en_a[i+1] = x_en_b[i+1]+d_ob_en[i+1]@K_oi.T
            x_hat_a[i + 1] = x_en_a[i + 1] - np.mean(x_en_a[i + 1], axis=0)
            P_a[i + 1] = 1 / (en_size - 1) * (x_hat_a[i + 1].T @ x_hat_a[i + 1])
            d_ab[i + 1] = x_a[i + 1] @ H_op.T - x_b[i + 1] @ H_op.T
            d_oa[i + 1] = y_o[i + 1] - x_a[i + 1] @ H_op.T
        return x_a[1:],x_b_non_DA[1:],d_ob[1:],d_ab[1:],d_oa[1:],P_b,P_a


