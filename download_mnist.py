import numpy as np
from urllib import request
import gzip
import pickle

filename = [
["training_images","mnist/train-images-idx3-ubyte.gz"],
["test_images","mnist/t10k-images-idx3-ubyte.gz"],
["training_labels","mnist/train-labels-idx1-ubyte.gz"],
["test_labels","mnist/t10k-labels-idx1-ubyte.gz"]
]

# def download_mnist():
#     base_url = "http://yann.lecun.com/exdb/mnist/"
#     for name in filename:
#         print("Downloading "+name[1]+"...")
#         request.urlretrieve(base_url+name[1], name[1])
#     print("Download complete.")

def save_mnist():
    mnist = {}
    # images
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
            print(name[0])
    #labels
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
            print(name[0])
    with open("mnist.pkl", 'wb') as f:
        print(mnist)
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    #download_mnist()
    save_mnist()
    #print((load()[0]).shape)
def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()
