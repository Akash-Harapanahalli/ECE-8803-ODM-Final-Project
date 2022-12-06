from numpy import array, eye, clip, copy, pi
from LinearSystem import LinearSystem

'''
x = [
    px, (position x)
    py, (position y)
    pz, (position z)
    vx, (velocity x)
    vy, (velocity y)
    vz  (velocity x) 
]
u = [
    tan(θ), (tan(pitch))
    tan(φ), (tan(roll))
    τ - g,  (net thrust with gravity accounted for)
]
'''

g = 9.8

class QuadrotorSystem (LinearSystem) :
    def __init__(self, Δ=0.01) -> None:
        cA = array([
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
        ])
        cB = array([
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [g,0,0],
            [0,-g,0],
            [0,0,1],
        ])
        super().__init__(eye(6) + Δ*cA, Δ*cB)

    def f (self, x, u) :
        cu = copy(u)
        # cu[0] = clip(cu[0],-pi/9,pi/9)
        # cu[1] = clip(cu[1],-pi/9,pi/9)
        # cu[2] = clip(cu[2],-2*g,2*g)
        return self.A@x + self.B@cu
