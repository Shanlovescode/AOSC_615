import numpy as np
from numba import njit


@njit()
def rk4(x, dxdt, dt, params):
    k1 = dxdt(x, params)
    k2 = dxdt(x + k1 / 2 * dt, params)
    k3 = dxdt(x + k2 / 2 * dt, params)
    k4 = dxdt(x + dt * k3, params)

    xnext = x + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return xnext
@njit()
def rk4_dxPdt(x,P,dxPdt,dxdt,FTLM,Qb,dt,params):
    k1x, k1p = dxPdt(x,P,dxdt,FTLM,params,Qb)
    k2x, k2p = dxPdt(x + k1x/2 * dt,P + k1p / 2 * dt,dxdt,FTLM,params,Qb)
    k3x, k3p = dxPdt(x + k2x/2 * dt,P + k2p / 2 * dt,dxdt,FTLM,params,Qb)
    k4x, k4p = dxPdt(x + dt*k3x, P + dt * k3p, dxdt,FTLM,params,Qb)
    xnext= x + 1 / 6 * dt * (k1x + 2 * k2x + 2 * k3x + k4x)
    Pnext = P + 1 / 6 * dt * (k1p + 2 * k2p + 2 * k3p + k4p)
    return xnext, Pnext


@njit()
def rk4_dxMdt(x,M,dxMdt,dxdt,FTLM,dt,params):
    k1x, k1m = dxMdt(x,M,dxdt,FTLM,params)
    k2x, k2m = dxMdt(x + k1x/2 * dt,M + k1m / 2 * dt,dxdt,FTLM,params)
    k3x, k3m = dxMdt(x + k2x/2 * dt,M + k2m / 2 * dt,dxdt,FTLM,params)
    k4x, k4m = dxMdt(x + dt*k3x, M + dt * k3m, dxdt,FTLM,params)
    xnext = x + 1 / 6 * dt * (k1x + 2 * k2x + 2 * k3x + k4x)
    Mnext = M+ 1 / 6 * dt * (k1m + 2 * k2m + 2 * k3m + k4m)
    return xnext,Mnext


@njit()
def run_model(x, T, dxdt, params, dt, discard_len,int_steps):
    output=np.zeros((T+discard_len,x.size))
    output[0]=x
    for i in range(T+discard_len):
        output[i+1] = run_step(output[i],dxdt,dt,int_steps,params)
    return output[discard_len:]

@njit(parallel = False)
def run_array(ic_array, dxdt, int_steps, h, params):
    model_output = np.zeros(ic_array.shape)
    for i in range(ic_array.shape[0]):
        model_output[i] = run_step(ic_array[i], dxdt, int_steps, h, params)
    return model_output

@njit()
def run_step(x,dxdt,dt,int_steps,params):
    for i in range(int_steps):
        x = rk4(x, dxdt, dt, params)
    return x

@njit()
def lorentz3_dxdt(x, params):
    return np.array([params[0] * (- x[0] + x[1]),
                     params[1] * x[0] - x[1] - x[0] * x[2],
                     x[0] * x[1] - params[2] * x[2]])
@njit()
def lorentz3_TLM(x, params):
    return np.array([[-params[0],params[0],0.0],
                     [params[1]-x[2],-1.0,-x[0]],
                     [x[1],x[0],- params[2]]])

@njit()
def general_dxPdt(x,P,dxdt,FTLM,params,Qb):
    F=FTLM(x,params)
    dxdtout=dxdt(x,params)
    dpdtout =F @ P + P.T @ F.T + Qb
    return dxdtout, dpdtout

@njit()
def general_dxMdt(x,M,dxdt,FTLM,params):
    F=FTLM(x,params)
    dxdtout=dxdt(x,params)
    dmdtout=F @ M
    return dxdtout,dmdtout
@njit()
def run_step_EK(x,P,dxPdt,dxdt,FTLM,dt,int_steps,params,Qb):
    for i in range(int_steps):
       x, P = rk4_dxPdt(x,P,dxPdt,dxdt,FTLM,Qb,dt,params)
    return x,P

@njit()
def run_step_TLM(x,M,dxdt,dxMdt,FTLM,dt,int_steps,params):
    for i in range(int_steps):
        x,M = rk4_dxMdt(x,M,dxMdt,dxdt,FTLM,dt,params)
    return x,M

@njit()
def run_step_non_linear(x,dx,dxdt,dxMdt,FTLM,dt,int_steps,params):
    for i in range(int_steps):
        x, dx = rk4_dxMdt(x, dx, dxMdt, dxdt, FTLM, dt, params)
    return x, dx