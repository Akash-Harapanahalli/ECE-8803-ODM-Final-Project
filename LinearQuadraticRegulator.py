import numpy as np
from LinearSystem import LinearSystem

# Infinite Horizon LQR
class LinearQuadraticRegulator :
    def __init__(self, sys:LinearSystem, Q=None, R=None, S0=None, Shor=10000) -> None:
        self.sys = sys
        self.Q = np.eye(sys.A.shape[0]) if Q is None else Q
        self.R = np.eye(sys.B.shape[1]) if R is None else R

        self.S = np.eye(sys.A.shape[0]) if S0 is None else S0
        for t in range(Shor) :
            self.S = self.Q + sys.A.T @ self.S @ sys.A \
                - sys.A.T @ self.S @ sys.B @ \
                    np.linalg.pinv(self.R + sys.B.T @ self.S @ sys.B) @ sys.B.T @ self.S @ sys.A
        self.K = -np.linalg.pinv(self.R + sys.B.T @ self.S @ sys.B) @ sys.B.T @ self.S @ sys.A

    def u(self, x) :
        return self.K @ x

# class RobustLinearQuadraticRegulator (LinearQuadraticRegulator) :
#     def __init__(self, sys: LinearSystem, w_min, w_max, Q=None, R=None, S0=None, Shor=10000) -> None:
#         super().__init__(sys, Q, R, S0, Shor)
    
#     def u(self, x):
#         u_pre = self.K @ x
#         u_post = np.empty_like(u_pre)
#         for i in len(u_post) :
#             u_post[i] = u_pre[i] + w_max
