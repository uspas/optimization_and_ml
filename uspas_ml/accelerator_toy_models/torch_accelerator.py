import numpy as np
import torch
from torch.nn import Module

class TorchAccelerator(Module):
    def __init__(self, elements):
        Module.__init__(self)
        
        self.elements = elements

        for ele in self.elements:
            self.add_module(ele.name, ele)

    def calculate_transport(self):
        M = torch.eye(6)
        for ele in self.elements:
            M = torch.matmul(ele(), M)
        return M
            
    def propagate_beam_matrix(self, beam_matrix, noise_level = 0.0):
        '''
        Propagates 6D beam matrix (sigma matrix)
        If noise level is non-zero we add Gaussian noise to the final beam matrix measurement 
        centered at zero with an rms of <noise_level>.
        
        '''        
        M = self.calculate_transport()
        return M @ beam_matrix @ torch.transpose(M,0,1) + noise_level**2 * torch.randn(*M.shape)
        

def rot(alpha):
    M = torch.eye(6)

    C = torch.cos(torch.tensor(alpha))
    S = torch.sin(torch.tensor(alpha))
    M[0,0] = C
    M[0,2] = S
    M[1,1] = C
    M[1,3] = S
    M[2,0] = -S
    M[2,2] = C
    M[3,1] = -S
    M[3,3] = C

    return M
        

        
class TorchQuad(Module):
    def __init__(self, name, L, K1, alpha = torch.tensor(0.0)):
        Module.__init__(self)
        #self.register_parameter('L',torch.nn.parameter.Parameter(length))
        #self.register_parameter('K1',torch.nn.parameter.Parameter(K1))
        self.L = L
        self.K1 = K1
        
        self.name = name
        
    def forward(self):
        M = torch.eye(6)

        if self.K1 < 0:
            K1 = -self.K1
            flip = True
        else:
            K1 = self.K1
            flip = False
        
        k = torch.sqrt(K1)
        
        kl = self.L * k
        M[0,0] = torch.cos(kl)
        M[0,1] = torch.sin(kl) / k
        M[1,0] = -k * torch.sin(kl)
        M[1,1] = torch.cos(kl)

        M[2,2] = torch.cosh(kl)
        M[2,3] = torch.sinh(kl) / k
        M[3,2] = k * torch.sinh(kl)
        M[3,3] = torch.cosh(kl)

        if flip:
            M = rot(- np.pi / 2) @ M @ rot(np.pi / 2)

        return M
        
class TorchDrift(Module):
    def __init__(self, name, length):
        Module.__init__(self)
        self.name = name
        self.register_parameter('L', torch.nn.parameter.Parameter(length))

    def forward(self):
        M = torch.eye(6)
        M[0,1] = self.L
        M[2,3] = self.L

        return M

        
