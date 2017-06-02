#%reset -f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
from numpy.linalg import inv
import importlib as imp  # e.g) imp.reload(gu)

import kernel_functions as kf
import gp_utils as gu
# exec(open("gp_utils.py").read())

plt.ion()

# -----------------------------------------------------------------------------


def true_func(x):
    return np.sin(np.pi * x)

x = np.linspace(1, 7, 1000)
fx = true_func(x)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(x, fx, 'orange')
plt.grid(alpha=0.2)
plt.ylim((-3, 3))
plt.xlabel(r'$time$')
plt.ylabel(r'$f(x)$')

covmat_prior = kf.kfunc_gauss(x, x, l=1)
plt.subplot(122)
gu.gp_plot(np.nan, np.nan, x, np.zeros_like(x), covmat_prior,
           n_sample=5, main="Samples from a prior")
plt.plot(x, fx, color='orange')
plt.xlabel(r'$time$')
plt.ylabel(r'$f(x)$')

# Sampling from the prior
# input observaitons
x_obs = np.linspace(1, 5, 30)
# noisy output obvervations
sig_noise = np.sqrt(0.05)
y_obs = true_func(x_obs) + np.random.normal(0, sig_noise, len(x_obs))
plt.plot(x_obs, y_obs, 'o')
plt.plot(x, true_func(x), color='orange')
plt.grid(alpha=0.2)

x_pred = x.copy()

# gauss
kernel_used = kf.kfunc_gauss
params = {'l': 0.5}
xo = x_obs[0:2]
yo = y_obs[0:2]
mu, var = gu.gp_solve(xo, yo, x_pred, kernel_used, sig_noise, **params)

plt.figure()
gu.gp_plot(xo, yo, x_pred, mu, var)
plt.plot(x, fx, 'orange')

input("")

for i in range(3, len(x_obs) + 1):
    plt.clf()
    xo = x_obs[0:i]
    yo = y_obs[0:i]
    mu, var = gu.gp_solve(xo, yo, x_pred, kernel_used, sig_noise, **params)
    gu.gp_plot(xo, yo, x_pred, mu, var)
    plt.plot(x, fx, 'orange', alpha=0.5)
    plt.pause(0.1)


# periodicic
kernel_used = kf.kfunc_per
params = {'l': 1, 'p': 2}
xo = x_obs[0:2]
yo = y_obs[0:2]
mu, var = gu.gp_solve(xo, yo, x_pred, kernel_used, sig_noise, **params)

plt.figure()
gu.gp_plot(xo, yo, x_pred, mu, var)
plt.plot(x, fx, 'orange')

input("")

for i in range(3, len(x_obs) + 1):
    plt.clf()
    xo = x_obs[0:i]
    yo = y_obs[0:i]
    mu, var = gu.gp_solve(xo, yo, x_pred, kernel_used, sig_noise, **params)

    gu.gp_plot(xo, yo, x_pred, mu, var)
    plt.plot(x, fx, 'orange', alpha=0.5)
    plt.pause(0.1)


# ----------------------
# Real Exmaple

#tmp = pd.read_csv('shop.csv')
tmp = pd.read_csv('auto.csv')
tmp = tmp.sort_index(ascending=False)

dt_raw = pd.Series(tmp['Close'].values, index=pd.to_datetime(tmp['Date']))
dt_smth = pd.ewma(dt_raw, halflife=15)
dt = dt_smth.resample('W-MON', how='last')
#dt = dt_smth.resample('M', how='last')
plt.figure()
plt.plot(dt, color='orange')
plt.grid(alpha=0.2)

t = np.arange(1, len(dt) + 1)
no_sample = 2
sig_noise = np.sqrt(0.01)

# periodicic
kernel_used = kf.kfunc_per
params = {'l': 5, 'p': 12 * 4}
t0 = t[0:no_sample]
y0 = dt[0:no_sample]
mu, var = gu.gp_solve(t0, y0, t, kernel_used, sig_noise, **params)

plt.figure()
gu.gp_plot(t0, y0, t, mu, var)

input("")

for i in range(no_sample + 1, len(t) + 1):
    plt.clf()
    t0 = t[0:i]
    y0 = dt[0:i]
    mu, var = gu.gp_solve(t0, y0, t, kernel_used, sig_noise, **params)
    gu.gp_plot(t0, y0, t, mu, var)
    plt.plot(t, dt, color='orange')
    plt.pause(0.001)

# linear
kernel_used = kf.kfunc_lin
params = {'b': 1, 'v': 0.1, 'c': 0}
t0 = t[0:no_sample]
y0 = dt[0:no_sample]
mu, var = gu.gp_solve(t0, y0, t, kernel_used, sig_noise, **params)

plt.figure()
gu.gp_plot(t0, y0, t, mu, var)

input("")

for i in range(no_sample + 1, len(t) + 1):
    plt.clf()
    t0 = t[0:i]
    y0 = dt[0:i]
    mu, var = gu.gp_solve(t0, y0, t, kernel_used, sig_noise, **params)
    gu.gp_plot(t0, y0, t, mu, var)
    plt.plot(t, dt, color='orange')
    plt.pause(0.001)

## periodicic + linear
kernel_used = kf.kfunc_per_add_lin
params = {'h': 1, 'l': 5, 'p': 12 * 4.4, 'b': 1, 'v': 0.1, 'c': 0}
t0 = t[0:no_sample]
y0 = dt[0:no_sample]
mu, var = gu.gp_solve(t0, y0, t, kernel_used, sig_noise, **params)

plt.figure()
gu.gp_plot(t0, y0, t, mu, var)

input("")

for i in range(no_sample + 1, len(t) + 1):
    plt.clf()
    t0 = t[0:i]
    y0 = dt[0:i]
    mu, var = gu.gp_solve(t0, y0, t, kernel_used, sig_noise, **params)
    gu.gp_plot(t0, y0, t, mu, var)
    plt.plot(t, dt, color='orange')
    plt.pause(0.001)
