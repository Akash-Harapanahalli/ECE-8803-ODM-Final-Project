import numpy as np
import matplotlib.pyplot as plt
from DoubleIntegratorSystem import DoubleIntegratorSystem
from LinearQuadraticRegulator import LinearQuadraticRegulator
from GradientPerturbationController import GradientPerturbationLQR
from Simulator import *

Δ = 0.01
T = 100
tt = np.linspace(0,Δ*T,T)

experiments = [
    ['const 0', 'gauss 0'],
    # ['const 1', 'gauss 1'],
    ['const 2', 'gauss 2'],
    ]
def make_dist (dist) :
    ds = dist.split()
    if ds[0] == 'none' :
        return None
    if ds[0] == 'gauss' :
        return [
            GaussianDisturbance([0,0], [[1e-3,0],[0,0.1]]),
            GaussianDisturbance([0,-0.5], [[1e-3,0],[0,0.1]]),
            GaussianDisturbance([0,-1], [[1e-3,0],[0,0.1]]),
        ][int(ds[1])]
    if ds[0] == 'const' :
        return [
            ConstantDisturbance(np.array([[0],[0]])),
            ConstantDisturbance(np.array([[0],[-0.5]])),
            ConstantDisturbance(np.array([[0],[-1]])),
        ][int(ds[1])]

Q = np.array([[1,0],[0,0.01]])
R = np.array([[0.05]])
sys = DoubleIntegratorSystem(Δ)

methods = ['LQR', 'GPC-LQR']
def make_method (method) :
    if method == 'LQR' :
        return LinearQuadraticRegulator(sys, Q, R)
    if method == 'GPC-LQR' :
        return GradientPerturbationLQR(sys, 8, 0.1, Q, R)

x0 = np.array([[1],[0]])

fig, axs = plt.subplots(len(experiments),len(experiments[0]), dpi=100, figsize=[9,6], squeeze=False)
fig.subplots_adjust(left=0.075,right=0.975,bottom=0.1,top=0.95, hspace=0.2, wspace=0.2)

for e_i,E in enumerate(experiments) :
    for e_j,dist in enumerate(E) :
        for method in methods :
            d = make_dist(dist)
            m = make_method(method)
            sim = Simulator(sys, m.u)
            x = sim.simulate(x0, T, d)

            axs[e_i,e_j].plot(tt,x[0,:], label=method)
            axs[e_i,e_j].legend()
        axs[e_i,e_j].plot([0,Δ*T],[0,0],ls='--')
        if dist.split()[0] == 'gauss' :
            mean = np.array2string(d.mvn.mean,separator=',')
            cov = np.array2string(np.diag(d.mvn.cov),separator=',')
            axs[e_i,e_j].set_title(f'$w_t\sim\mathcal{{N}}({mean},$diag$({cov}))$')
        if dist.split()[0] == 'const' :
            str = np.array2string(d.w().reshape(-1),separator=',')
            axs[e_i,e_j].set_title(f'$w_t={str}$')
        axs[e_i,e_j].set_ylim([-0.5,1.5])
        axs[e_i,e_j].set_ylabel('$x_1$',rotation=0)
        if e_i == len(experiments) - 1 :
            axs[e_i,e_j].set_xlabel('t')
plt.savefig('figures/double_integrator/experiments.pdf')
plt.show()