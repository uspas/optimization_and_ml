import numpy as np
import torch

from . import torch_accelerator

def create_singlet(K):
    '''
        creates a 1st order matrix model of quadrupole triplet with 
        tunable geometric strength parameters

        Quads are seperated by 1m and distance from last quad to screen is 0.5m
        Quads have a thickness of 10 cm

        Arguments
        ---------

        K : torch.tensor, shape = (1,1)
            Geometric quadrupole strengths (units: 1/m**2)

    '''

    #define constants
    quad_thick = torch.tensor(0.01)
    quad_sep = torch.tensor(1.0)
    quad_to_screen = torch.tensor(1.0)

    #define accelerator components
    Q1 = torch_accelerator.TorchQuad('Q1', quad_thick, K)
    
    D1 = torch_accelerator.TorchDrift('D1', torch.tensor(1.0))

    #define beamline
    LINE = torch_accelerator.TorchAccelerator([D1,Q1,D1])
    
    return LINE



def create_triplet(K):
    '''
        creates a 1st order matrix model of quadrupole triplet with 
        tunable geometric strength parameters

        Quads are seperated by 1m and distance from last quad to screen is 0.5m
        Quads have a thickness of 10 cm

        Arguments
        ---------

        K : torch.tensor, shape = (3,)
            Geometric quadrupole strengths (units: 1/m**2)

    '''


    #define constants
    quad_thick = torch.tensor(0.1)
    quad_sep = torch.tensor(1.0)
    quad_to_screen = torch.tensor(0.5)

    #define accelerator components
    Q1 = torch_accelerator.TorchQuad('Q1', quad_thick, K[0])
    Q2 = torch_accelerator.TorchQuad('Q2', quad_thick, K[1])
    Q3 = torch_accelerator.TorchQuad('Q3', quad_thick, K[2])

    D1 = torch_accelerator.TorchDrift('D1', quad_sep)
    D2 = torch_accelerator.TorchDrift('D2', torch.tensor(0.5))

    #define beamline
    LINE = torch_accelerator.TorchAccelerator([D1,Q1,D1,Q2,D1,Q3,D2])
    
    #prevent quad lengths and drift lengths from being trained
    #ONLY needed when using torch.optim functions otherwise ignore 
    trainable = ['Q1.K1', 'Q2.K1', 'Q3.K1']
    for name, item in LINE.named_parameters():
        if not (name in trainable):
            item.requires_grad = False

    return LINE
        
        

        
