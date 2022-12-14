import numpy as np
from LinearSystem import LinearSystem
from LinearQuadraticRegulator import LinearQuadraticRegulator
import time

# Gradient Perturbation Controller for inf horizon LQR cost formulation
class GradientPerturbationLQR (LinearQuadraticRegulator) :
    def __init__(self, sys:LinearSystem, h:int=8, η:float=0.025, Q=None, R=None, M=None, S0=None, Shor=10000) -> None:
        super().__init__(sys, Q, R, S0, Shor)
        self.h = h
        self.M = np.array([np.zeros_like(sys.B.T,dtype=np.float64) for _ in range(self.h)]) if M is None else M
        self.M0 = np.zeros((sys.B.shape[1],1))
        self.w = np.array([np.zeros((sys.A.shape[0],1)) for _ in range(2*self.h+1)])
        self.x = np.array([np.zeros((sys.A.shape[0],1)) for _ in range(self.h)])
        self._u = None
        self.init = True
        self.c = lambda x, u : x.T @ self.Q @ x + u.T @ self.R @ u
        self.η = η
    
    def u(self,x) :
        if self.init :
            self.init = False
            self.x[0,:] = x
            self._u = self.K @ x + np.sum(self.M @ self.w[:self.h], axis=0) + self.M0
            return self._u
        self.x = np.roll(self.x, 1, axis=0)
        self.x[0,:] = x 
        self.w = np.roll(self.w, 1, axis=0)
        self.w[0,:] = x - self.sys.f(self.x[1,:], self._u)
        gM, gM0 = self.gl()
        self.M -= self.η * gM
        self.M0 -= self.η * gM0
        self._u = self.K @ x + np.sum(self.M @ self.w[:self.h], axis=0) + self.M0
        return self._u
    
    def l(self, M, M0) :
        x = self.x[self.h-1,:]
        for t in range(self.h) :
            u = self.K @ x + np.sum(M @ self.w[self.h-t:2*self.h-t], axis=0) + M0
            x = self.sys.f(x, u) + self.w[self.h-1-t]
        return self.c(x,u)

    def gl (self, eps=0.0001) :
        gM = np.empty_like(self.M)
        gM0 = np.empty_like(self.M0)
        M = np.copy(self.M)
        M0 = np.copy(self.M0)
        L = self.l(self.M,self.M0)
        for idx in np.ndindex(self.M.shape) :
            M[idx] += eps
            gM[idx] = (self.l(M,M0) - L) / eps
            M[idx] -= eps
        for idx in np.ndindex(self.M0.shape) :
            M0[idx] += eps
            gM0[idx] = (self.l(M,M0) - L) / eps
            M0[idx] -= eps
        return gM, gM0