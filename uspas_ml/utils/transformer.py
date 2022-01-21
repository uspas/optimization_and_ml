import numpy as np
import torch

'''lightweight class for normalization/standardization using numpy'''


class Transformer:
    def __init__(self, x, transform_type = 'unitary'):
        '''
        Transformer class that allows normalization and standardization of parameters.
        - Use forward method to normalize input vector
        - Use backward method to unnormalize input vector
        Does not support backpropagation!
        
        Arguments
        ---------
        x : ndarray, shape (N x M), optional, default None
             Input data to determine normalization parameters where N is the number of points and M is the dimensionality
        
        bounds : ndarray, shape (M x 2), optional, default None
             Alternate specification of normalization bounds instead of data, bounds[M][0] is the M'th lower bound,
                                                                              bounds[M][1] is the M'th upper bound
        
        transform_type : ['unitary', 'normalize', standardize']
            Transformation method.
                - 'unitary' : No modification of input data
                - 'normalize' : Scales and shifts data s.t. data is between 0 and 1
                - 'standardize' : Scales and shifts data s.t. data has a mean of 0.0 and a rms size of 1.0
        
        
        '''
        
        possible_transformations = ['unitary','normalize','standardize']
        assert transform_type in possible_transformations
        
        self.ttype = transform_type
       
        assert len(x.shape) == 2
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
            
        self.x = x

        self._get_stats()
        
    def _get_stats(self):
        if self.ttype == 'normalize':
            self.mins = np.min(self.x, axis = 0)
            self.maxs = np.max(self.x, axis = 0) 

        elif self.ttype == 'standardize':
            self.means = np.mean(self.x, axis = 0)
            self.stds = np.std(self.x, axis = 0)

    def recalculate(self, x):
        #change transformer data and recalculate stats
        self.x = x
        self._get_stats()
    
    def forward(self, x_old):
        #if x_old is a torch tensor get numpy array from it
        if isinstance(x_old, torch.Tensor):
            x_old = x_old.detach().numpy()
            torch_input = True
        
        x = x_old.copy()
        assert len(x.shape) == 2

        
        if self.ttype == 'normalize':
            for i in range(x.shape[1]):
                if self.maxs[i] - self.mins[i] == 0.0:
                    x[:,i] = x[:,i] - self.mins[i]
                else:
                    x[:,i] = (x[:,i] - self.mins[i]) /(self.maxs[i] - self.mins[i])
                    
        elif self.ttype == 'standardize':
            for i in range(x.shape[1]):
                if self.stds[i] == 0:
                    x[:,i] = x[:,i] - self.means[i]
                else:
                    x[:,i] = (x[:,i] - self.means[i]) / self.stds[i]

        if torch_input:
            x = torch.from_numpy(x)
        
        return x
                
    def backward(self, x_old):
        
        #if x_old is a torch tensor get numpy array from it
        if isinstance(x_old, torch.Tensor):
            x_old = x_old.detach().numpy()
            torch_input = True

        x = x_old.copy()
        assert len(x.shape) == 2
        
        if self.ttype == 'normalize':
            for i in range(x.shape[1]):
                x[:,i] = x[:,i] * (self.maxs[i] - self.mins[i]) + self.mins[i]

        elif self.ttype == 'standardize':
            for i in range(x.shape[1]):
                x[:,i] = x[:,i] * self.stds[i] + self.means[i]
    
        if torch_input:
            x = torch.from_numpy(x)
            
        return x
                
if __name__ == '__main__':
    #testing suite
    x = np.random.uniform(size = (10,2)) * 10.0
    print(x)
    t = Transformer(x, 'standardize')
    x_new = t.forward(x)
    print(x_new)
    print(t.backward(t.forward(x)))
