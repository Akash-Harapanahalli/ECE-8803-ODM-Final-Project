import numpy as np

class LinearSystem :
    def __init__(self, A, B) -> None:
        self.A = A
        self.B = B
    def f (self, x, u) :
        return self.A@x + self.B@u

class DoubleIntegrator (LinearSystem) :
    def __init__(self, Δ=0.01) -> None:
        super().__init__(np.array([[1,Δ],[0,1]]), np.array([[0],[1]]))
