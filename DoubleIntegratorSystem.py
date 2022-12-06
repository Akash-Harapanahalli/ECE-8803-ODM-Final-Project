from numpy import array
from LinearSystem import LinearSystem

class DoubleIntegratorSystem (LinearSystem) :
    def __init__(self, Δ=0.01) -> None:
        super().__init__(array([[1,Δ],[0,1]]), array([[0],[1]]))
