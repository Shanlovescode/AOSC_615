from src.model_functions import *
import numpy as np
from numba import njit

class Lorentz3():
    def __init__(self, dt=0.01, int_steps=10, sigma=10.,beta=8 / 3, rho=28., ic=np.array([]), ic_seed=0,obs_noise=1e-5,noise_seed=10):
        self.params = np.array([sigma, rho, beta])
        self.dxdt = lorentz3_dxdt
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
        return output + self.obs_noise* rng.normal(size=output.shape)

    def observe(self, H_op, x_o):
        return x_o @ H_op.T


