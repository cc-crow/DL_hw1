import numpy as np
import cupy as cp

#线性层
class linear(object):
    def __init__(self,num_input,num_output):
        self.weight = cp.random.normal(loc=0, scale=0.01, size=(num_input,num_output))
        self.bias = cp.zeros((1,num_output))

    def forward(self,X):
        Y = X@self.weight+self.bias
        return Y
    
    def __call__(self,X):
        return self.forward(X)
        
    def parameters(self):
        return [self.weight,self.bias]
        
    def load_state_dict(self,param):
        self.weight = param[0]
        self.bias = param[1]
    
    def cpu(self):
        self.weight = self.weight.get()
        self.bias = self.bias.get()
        
    def cuda(self):
        self.weight = cp.asarray(self.weight)
        self.bias = cp.asarray(self.bias)
        
#激活函数relu
class relu(object):
    def __init__(self):
        pass
       
    def forward(self,X):
        return (cp.abs(X)+X)/2
    
    def __call__(self,X):
        return self.forward(X) 

#两层mlp  
class mlp(object):
    def __init__(self,num_input,num_hidden,num_output,lr,l2,milestone=500,gamma=0.5):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.fc1 = linear(num_input,num_hidden)
        self.relu = relu()
        self.fc2 = linear(num_hidden,num_output)
        self.H = 0
        self.Z = 0
        self.K = 0
        
        self.lr = lr
        #学习率下降
        self.milestone = milestone
        self.gamma = gamma
        self.lr_ = lr
        self.l2 = l2
        
    def forward(self,X):
        Y = self.fc1(X)
        self.H = Y
        Y = self.relu(Y)
        self.Z = Y
        Y = self.fc2(Y)
        self.K = cp.exp(Y)
        return Y
        
    def __call__(self,X):
        return self.forward(X)
        
    def parameters(self):
        return self.fc1.parameters()+self.fc2.parameters()
        
    def load_state_dict(self,param):
        self.fc1.load_state_dict(param[:2])
        self.fc2.load_state_dict(param[2:])
        
    def backward(self,X,y):
        grad = [0]*4
        for i in range(len(X)):
            x = X[i]
            z = self.Z[i]
            k = self.K[i]
            k_diag = cp.diag(k)
            h = self.H[i]
            h_diag = cp.diag(np.where(h>0,1.,0.))
            y_hot = cp.eye(10)[y[i]]
            e = cp.ones((10,1))
            df = 1/(k@e)*e.T-1/(k@y_hot.T)*y_hot
            dk = k_diag@self.fc2.weight.T
            #w1的梯度
            g = []
            for j in range(len(x)):
                g.append(x[j]*h_diag)
            g = cp.hstack(g)
            grad[0] += (df@dk@g).T.reshape(self.num_input,self.num_hidden)
            #b1的梯度
            grad[1] += df@dk@h_diag
            #w2的梯度
            g = []
            for j in range(len(z)):
                g.append(z[j]*k_diag)
            g = cp.hstack(g)
            grad[2] += (g.T@df.T).reshape(self.num_hidden,self.num_output)
            #b2的梯度
            grad[3] += df@k_diag
        for i in range(4):
            grad[i] = grad[i]/len(X)
            
        grad[0] = grad[0]+2*self.l2*self.fc1.weight
        grad[2] = grad[2]+2*self.l2*self.fc2.weight
        return grad
    
    #梯度下降
    def step(self,grad):
        self.fc1.weight -= self.lr_*grad[0]
        self.fc1.bias -= self.lr_*grad[1]
        self.fc2.weight -= self.lr_*grad[2]
        self.fc2.bias -= self.lr_*grad[3]
    
    #学习率衰减
    def lr_decay(self,epoch):
        n = int(epoch/self.milestone)
        self.lr_ = self.lr*(self.gamma**n)
        
    def cpu(self):
        self.fc1.cpu()
        self.fc2.cpu()
        
    def cuda(self):
        self.fc1.cuda()
        self.fc2.cuda()
        