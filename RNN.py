# text classification (positive/negative)

import numpy as np

def oneHotEncoding(z):
    data = []
    l=[]
    for i in z:
        l = i.split()
        data = data + l
    data = list(set(data))
    data = np.array(data)
    categories = np.unique(data)
    category_to_index = {category: index for index, category in enumerate(categories)}
    one_hot_encoded = np.zeros((data.size, categories.size))
    for i, category in enumerate(data):
        one_hot_encoded[i, category_to_index[category]] = 1
    return one_hot_encoded

def tanh(x):
    return np.tanh(x)

def softMax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class RNN():
    def __init__(self, x):
        self.x = x
        self.wx = np.random.randn(x.shape[1],x.shape[1])
        self.wh = np.random.randn(x.shape[1],x.shape[1])
        self.wy = np.random.randn(2,x.shape[1])
        self.hs = np.zeros((x.shape[0],x.shape[1]))
    
    def forward(self):
        h = np.zeros(self.x.shape[1])
        for t in range (self.x.shape[0]):
            self.hs[t] = tanh(np.dot(self.wh, h) + np.dot(self.wx, self.x[t]))
            h = self.hs[t]
        y = softMax(np.dot(self.wy, self.hs[-1]))
        return y
    
rnn = RNN(oneHotEncoding(input("input:")))
output = rnn.forward()
if output[0] > output[1]:
    print ("positive")
else:
    print ("negative")
