import numpy as np

class LinearSystem :
    def __init__(self, A, B) -> None:
        self.A = A
        self.B = B
    def f (self, x, u) :
        return self.A@x + self.B@u
