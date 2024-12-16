from dataclasses import dataclass

import numpy as np
import numpy.linalg as LA


class LinearModel:

    def __init__(self, A0: np.ndarray, B0: np.ndarray, C0: np.ndarray):
        
        self.A = LA.solve(C0, A0)
        self.B = LA.solve(C0, B0)

    def x0(self, u0):
        I = np.eye(self.A.shape[0])
        return LA.solve(I - self.A, self.B @ u0)

    def simulate(self, u, x0):
        
        n = self.A.shape[0]

        T = len(u)
        x = np.empty((T, n))
        x[0] = x0

        for t in range(1, T):
            x[t] = self.A @ x[t-1] + self.B @ u[t]

        return x


@dataclass
class AdasParam:
    """Class of storing the parameters of AD-AS model"""
    alpha: float
    phi: float
    theta_pi: float
    theta_Y: float

    pi_star: float
    Y_bar: float
    rho: float

    def __repr__(self):
        line1 = f"AdasParam(alpha={self.alpha}, phi={self.phi}, theta_pi={self.theta_pi}, theta_Y={self.theta_Y},"
        line2 = f"          pi_star={self.pi_star}, Y_bar={self.Y_bar}, rho={self.rho})"
        return line1 + "\n" + line2
    

class ADAS(LinearModel):
    """Class of dynamic AD-AS model"""

    def __init__(self, param: AdasParam):
        self.param = param
        
        C0 = np.array([
            [0, 0, 1, 0, self.param.alpha],
            [0, 1, 0, -1, 1],
            [1, 0, -self.param.phi, 0, 0],
            [1, -1, 0, 0, 0],
            [-(1+self.param.theta_pi), 0, -self.param.theta_Y, 1, 0]
        ])

        A0 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1., 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        B0 = np.array([
            [0, 1, 0, 1, self.param.alpha],
            [0, 0, 0, 0, 0],
            [1, 0, 0, -self.param.phi, 0],
            [0, 0, 0, 0, 0],       
            [0, 0, -self.param.theta_pi, -self.param.theta_Y, 1],
        ])

        super().__init__(A0, B0, C0)


if __name__ == "__main__":

    param = AdasParam(
        alpha = 1.0,
        phi = 0.25,
        theta_pi = 0.5,
        theta_Y = 0.5,
        pi_star = 2.0,
        Y_bar = 100.,
        rho = 2.0
    )
    
    model = ADAS(param)

    u = np.zeros((13, 5))     # 1列目がインフレショック、2列目が需要ショック
    u[:, 2] = param.pi_star
    u[:, 3] = param.Y_bar
    u[:, 4] = param.rho

    x0 = model.x0(u[-1])
    u[2, 0] = 1.    

    print(model.simulate(u, x0))

