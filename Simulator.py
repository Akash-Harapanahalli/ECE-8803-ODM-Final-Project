import numpy as np
from scipy.stats import multivariate_normal

class Disturbance :
    def __init__(self) -> None:
        pass
    def w(self) :
        pass

class ConstantDisturbance (Disturbance) :
    def __init__(self, w) -> None:
        super().__init__()
        self.w_ = w
    def w(self) :
        return self.w_

class GaussianDisturbance (Disturbance) :
    def __init__(self, mean, cov, seed=1) -> None:
        super().__init__()
        self.mvn = multivariate_normal(mean, cov, seed=seed)
    def w(self) :
        return self.mvn.rvs(1).reshape(-1,1)

class Simulator :
    def __init__(self, sys, u) -> None:
        self.sys = sys
        self.u = u

    def simulate (self, x0, T, dis:Disturbance=None) :
        x = np.empty((len(x0),T))
        x[:,(0,)] = x0
        for t in range(T-1) :
            if dis is not None:
                x[:,(t+1,)] = self.sys.f(x[:,(t,)], self.u(x[:,(t,)])) + dis.w()
            else :
                x[:,(t+1,)] = self.sys.f(x[:,(t,)], self.u(x[:,(t,)]))
        return x

