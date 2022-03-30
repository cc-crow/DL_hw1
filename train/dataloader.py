import numpy as np
import cupy as cp

#生成一个数据迭代器，最后一个大小不足Batch_size的Batch将被丢弃
class DataLoader(object):
    def __init__(self,X,y,batch_size):
        self.X = X
        self.y = y
        self.length = len(y)
        self.arr = np.array(range(self.length))
        self.batch_size = batch_size
        
    def __iter__(self):
        self.num = 0
        self.seq = np.random.permutation(self.arr)
        return self
        
    def __next__(self):
        if self.num+self.batch_size <= self.length:
            sample = self.seq[self.num:(self.num+self.batch_size)]
            self.image = self.X[sample]
            self.label = self.y[sample]
            self.num += self.batch_size
            return self.image, self.label
        else:
            raise StopIteration
            
    def __len__(self):
        return len(self.y)