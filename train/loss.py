import numpy as np
import cupy as cp

def softmax(x):
    max_x = cp.max(x,axis=1).reshape(-1,1)
    x = cp.exp(x-max_x) / cp.sum(cp.exp(x-max_x), axis = 1, keepdims = True)
    return x

def CrossEntropyLoss(y_hat,y):
    y_hat = softmax(y_hat)
    l = 0
    for i,j in enumerate(y):
        l += -cp.log(y_hat[i,j]+1e-05)
    return l/len(y)
    
    