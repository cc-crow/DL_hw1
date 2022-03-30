import numpy as np
import cupy as cp
from loss import CrossEntropyLoss

def validate(iter_,net):
    accuracy = 0
    l = []
    for X,y in iter_:
        X = cp.asarray(X)
        y = cp.asarray(y)
        y_hat = net(X)
        loss = CrossEntropyLoss(y_hat,y)
        accuracy += (cp.argmax(y_hat,axis=1)==y).sum()
        l.append(loss.item())
    return accuracy/len(iter_), np.mean(l) 
        