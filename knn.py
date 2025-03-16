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

#L1 distance function
def d1(I1, I2): 
    if I1.shape != I2.shape:
        raise ValueError("L1 Error: vectors must have the same dimensions.")
    sum = 0 
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            sum += abs(I1[i,j] - I2[i,j])

    return sum

#L2 distance function
def d2(I1, I2):
    if I1.shape != I2.shape:
        raise ValueError("L2 Error: vectors must have the same dimensions.")
    sum = 0
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            sum += (I1[i,j] - I2[i,j])**2
    
    return math.sqrt(sum)

def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    ########################
    # Input your code here #
    ########################
    for input in newInput: 
        distances = []
        for i in range(dataSet.shape[0]): 
            #set distance metric here
            distance = d2(input, dataSet[i])
            distances.append(distance)

        #returns k smallest distance indices 
        idxs = np.argpartition(distances, k)
        count = [0] * 10
        
        for idx in idxs[:k]: 
            count[labels[idx]] += 1

        result.append(np.argmax(count))


    
    ####################
    # End of your code #
    ####################
    return result

accuracies = []
for k in range(3,4): 
    start_time = time.time()
    outputlabels=kNNClassify(x_test[0:25],x_train[:2500],y_train[:2500], k)
    result = y_test[0:25] - outputlabels
    result = (1 - np.count_nonzero(result)/len(outputlabels))
    accuracies.append(result)
    print("---utilizing %s nearest neighbors ---" %k)
    print ("---classification accuracy for knn on mnist: %s ---" %result)
    print ("---execution time: %s seconds ---" % (time.time() - start_time))
print("Highest accuracy achieved with %s nearest neighbors" %np.argmax(accuracies))
