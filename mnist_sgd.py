import random

import pandas as pd
import numpy as np
from PIL import Image as im
from thinktank import init_random_network

df = pd.read_csv('data/mnist_train.csv')
npa = df.to_numpy()
np.random.shuffle(npa)
images = [np.array([list(x[1::])]).transpose() for x in npa]
labels = [np.array([[0,0,0,0,0,0,0,0,0,0]]).transpose() for i in range(len(images))]
for i in range(len(images)):
    labels[i][npa[i][0]] = 1
def show_image(imarr):
    data = im.fromarray(imarr.reshape((int(np.sqrt(len(imarr))),int(np.sqrt(len(imarr))))).astype('uint8'))
    data.show()


# def percent_correct()
# def down_sample(imarr):
#     # 24x24 -> 12x12
# approaching cos from a random initialised sigmoid nnetwork via sgd.
# sigmoid
active = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
dactive =np.vectorize(lambda x: (1/(1 + np.exp(-x))) * (1- (1/(1 + np.exp(-x)))))
# softmax on final layer
sftmax = lambda x:(1/np.sum(np.exp(x)))*np.exp(x)
# gives jacobian
def dsftmax(x):
    # x should be a col array
    jacob = np.zeros((x.shape[0], x.shape[0]))
    id = np.identity(x.shape[0])
    sf = sftmax(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            jacob[j,i] = sf[i] * (id[i,j] - sf[j])
    # print(jacob)
    return jacob

# log loss
loss = lambda y1,y2: (-1/y1.shape[0])*(np.multiply(y2,np.log(y1)).sum())
dloss = lambda y1,y2: (-1/y1.shape[0])*(np.multiply(y2,np.reciprocal(y1)))
def emp_risk(preddata, actualdata):
    # print([(x, y) for x, y in zip(preddata, actualdata)])
    return (1 / len(preddata)) * sum([loss(x, y) for x, y in zip(preddata, actualdata)])

net = init_random_network([784,300,200,10], active, dactive, loss, dloss, linear_on_final=False,momentum_coef=9/10)
net.layers[-1].acvfn = sftmax
net.layers[-1].deriv_acvfn = dsftmax
# batch size
m = 1

batchinputs = [images[i:i + m] for i in range(0, len(images), m)]
batchoutputs = [labels[i:i + m] for i in range(0, len(labels), m)]
def get_pred(x):
    pred = 0
    for i in range(len(x)):
        if x[i] > x[pred]:
            pred = i
    return pred

def percent_correct(predata, actualdata):
    k = 0
    for a,b in zip(predata,actualdata):
        k+=int()
# try:
totalcorrect = 0
for k in range(len(batchoutputs)):
    net.sgd(batchinputs[k], batchoutputs[k], 0.01)
    totalcorrect += int(get_pred(net.layers[-1].out) == get_pred(batchoutputs[k][-1]))
    if (k+1)%100 == 0:
        print(f"% correct at round {k+1}: {totalcorrect/(k+1)}")

# test
df = pd.read_csv('data/mnist_test.csv')
npa = df.to_numpy()
correct = 0
print("testing")
for im in npa:
    correct+=int(get_pred(net.compute(np.array([list(im[1::])]).transpose()))==im[0])

print(f"% correct on test data: {100*correct/len(npa)}")