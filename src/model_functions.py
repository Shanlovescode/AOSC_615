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
def rk4_dPdt(P,dPdt,F,Qb,dt,params):
    k1 = dPdt(P, F,params,Qb)
    k2 = dPdt(P + k1 / 2 * dt, F,params,Qb)
    k3 = dPdt(P + k2 / 2 * dt, F,params,Qb)
    k4 = dPdt(P + dt * k3, F,params,Qb)
    Pnext = P + 1 / 6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return Pnext

@njit()
def run_model(x, T, dxdt, params, dt, discard_len,int_steps):
    output=np.zeros((T+discard_len,x.size))
    output[0]=x
    for i in range(T+discard_len):
        output[i+1] = run_step(output[i],dxdt,dt,int_steps,params)
    return output[discard_len:]

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
def general_dPdt(P,F,params,Qb):
    dpdt=F @ P + P.T @ F.T + Qb
    return dpdt
@njit()
def run_step_EK(x,P,dxdt,dPdt,FTLM,dt,int_steps,params,Qb):
    for i in range(int_steps):
        F = FTLM(x, params)
        x = rk4(x, dxdt, dt, params)
        P = rk4_dPdt(P,dPdt,F,Qb,dt,params)
    return x,P

