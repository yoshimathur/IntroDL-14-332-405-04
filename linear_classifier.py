import math
import numpy as np  
from download_mnist import load
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

def compress(img): 
    v = []
    width = img.shape[0]
    height = img.shape[1]
    for i in range(width):
        for j in range(height):
            v.append(img[i][j]) 
    
    #append bias placeholder
    v.append(1)
    return np.array(v)

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred))
    loss = np.dot(y_true, np.log(y_pred)) 
    return loss
    
def train(x_train, y_train, epochs): 
    best_loss = float('inf')
    best_W = None
    for epoch in range(epochs): 
        W = np.random.randn(10, (28*28 + 1)) * 0.0001
        loss = 0
        for i, x in enumerate(x_train): 
            x = compress(x)
            y = np.zeros(10)
            y[y_train[i]] = 1
            pred = np.dot(W, x)
            sample_loss = cross_entropy_loss(y, pred)
            loss += sample_loss
        loss /= x_train.shape[0]
        loss *= -1
        if loss < best_loss: 
            best_loss = loss
            best_W = W
        print('In epoch %d the loss was %.4f, best %.4f' %(epoch, loss, best_loss))
    
    return best_W

def predict(x_test, W):
    result = []
    for x in x_test: 
        x = compress(x)
        pred = np.dot(W, x)
        result.append(np.argmax(pred))
    return result

start_time = time.time()
W = train(x_train[:2500], y_train[:2500], 100)
outputlabels = predict(x_test[:25], W)
result = y_test[0:25] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for linear classifier on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))