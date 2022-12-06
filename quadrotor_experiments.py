import numpy as np
import matplotlib.pyplot as plt
from QuadrotorSystem import QuadrotorSystem
from LinearQuadraticRegulator import LinearQuadraticRegulator
from GradientPerturbationController import GradientPerturbationLQR
from Simulator import *

Δ = 0.01
T = 400
tt = np.linspace(0,Δ*T,T)

experiments = [
    ['const 0', 'gauss 1']
    # ['const 0', 'const 1']
    # ['const 0', 'gauss 0'],
    # ['const 1', 'gauss 1'],
    # ['const 2', 'gauss 2'],
    ]
def make_dist (dist) :
    ds = dist.split()
    if ds[0] == 'none' :
        return None
    if ds[0] == 'gauss' :
        cov = 1e-3*np.eye(6)
        return [
            GaussianDisturbance([0,0,0,0,0,0], cov),
            GaussianDisturbance([0,5,0,0,0,0], cov),
        ][int(ds[1])]
    if ds[0] == 'const' :
        return [
            ConstantDisturbance(np.array([0,0,0,0,0,0]).reshape(-1,1)),
            ConstantDisturbance(np.array([0,5,0,0,0,0]).reshape(-1,1)),
        ][int(ds[1])]

Q = np.array([
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,0.001,0,0],
    [0,0,0,0,0.001,0],
    [0,0,0,0,0,0.001]
])
R = np.array([
    [1e-5,0,0],
    [0,1e-5,0],
    [0,0,1e-5]
])
sys = QuadrotorSystem(Δ)

methods = ['LQR', 'GPC-LQR']
def make_method (method) :
    if method == 'LQR' :
        return LinearQuadraticRegulator(sys, Q, R)
    if method == 'GPC-LQR' :
        return GradientPerturbationLQR(sys, 8, 5, Q, R)

x0 = np.array([5,5,5,0,0,0]).reshape(-1,1)

fig, axs = plt.subplots(len(experiments),len(experiments[0]), dpi=1000, figsize=[9,4.5], squeeze=False, subplot_kw=dict(projection='3d'))
# fig, axs = plt.subplots(len(experiments),len(experiments[0]), dpi=100, figsize=[9,6], squeeze=False)
fig.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)

for e_i,E in enumerate(experiments) :
    for e_j,dist in enumerate(E) :
        for method in methods :
            d = make_dist(dist)
            m = make_method(method)
            sim = Simulator(sys, m.u)
            x = sim.simulate(x0, T, d)

            axs[e_i,e_j].plot3D(x[0,:],x[1,:],x[2,:], label=method)
            # axs[e_i,e_j].plot(tt,x[0,:], label='x')
            # axs[e_i,e_j].plot(tt,x[1,:], label='y')
            # axs[e_i,e_j].plot(tt,x[2,:], label='z')
            axs[e_i,e_j].legend()
            axs[e_i,e_j].set_xlabel('$p_x$')
            axs[e_i,e_j].set_ylabel('$p_y$')
            axs[e_i,e_j].set_zlabel('$p_z$')
            axs[e_i,e_j].set_ylim(0,axs[e_i,e_j].get_ylim()[1])
        if dist.split()[0] == 'gauss' :
            mean = np.array2string(d.mvn.mean,separator=',')
            # cov = np.array2string(np.diag(d.mvn.cov),separator=',')
            cov = '0.001I_6'
            axs[e_i,e_j].set_title(f'$w_t\sim\mathcal{{N}}({mean},{cov})$')
        if dist.split()[0] == 'const' :
            str = np.array2string(d.w().reshape(-1),separator=',')
            axs[e_i,e_j].set_title(f'$w_t={str}$')
plt.savefig('figures/quadrotor/experiments.png')
plt.show()