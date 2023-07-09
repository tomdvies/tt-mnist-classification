import pandas as pd
import numpy as np
from PIL import Image as img
from thinktank import init_random_network
import time

# classification of mnist digits via a fcnn with sgd + momentum
df = pd.read_csv('data/mnist_train.csv')
npa = df.to_numpy()
np.random.shuffle(npa)
# was encountering overflow errors so scaling is to avoid that
images = [(1/256)*np.array([list(x[1::])]).transpose() for x in npa]
labels = [np.array([[0,0,0,0,0,0,0,0,0,0]]).transpose() for i in range(len(images))]
for i in range(len(images)):
    labels[i][npa[i][0]] = 1
def show_image(imarr):
    data = img.fromarray(imarr.reshape((int(np.sqrt(len(imarr))),int(np.sqrt(len(imarr))))).astype('uint8'))
    data.show()

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

net = init_random_network([784,400,300,10], active, dactive, loss, dloss, linear_on_final=False,momentum_coef=9/10)
net.layers[-1].acvfn = sftmax
net.layers[-1].deriv_acvfn = dsftmax

with open("tt-mnist-classification-weights","r") as f:
    wstr = f.read()

# net.load_network_from_string(wstr)
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

# try:
# totalcorrect = 0
batchloss = 0
m=500
for step_size in [0.1,0.1,0.05,0.05,0.05]:
    for k in range(len(batchoutputs)):
        net.sgd(batchinputs[k], batchoutputs[k], step_size)
        # totalcorrect += int(get_pred(net.layers[-1].out) == get_pred(batchoutputs[k][-1]))
        batchloss += loss(net.layers[-1].out,batchoutputs[k][-1])
        if (k+1)%m == 0:
            print(f"loss in batch {k+1}: {batchloss/m}")
            batchloss = 0

# totalcorrect = 0
# for k in range(len(batchoutputs)):
#     net.sgd(batchinputs[k], batchoutputs[k], 0.05)
#     totalcorrect += int(get_pred(net.layers[-1].out) == get_pred(batchoutputs[k][-1]))
#     if (k+1)%500 == 0:
#         print(f"% correct at step size 2, round {k+1}: {totalcorrect/(k+1)}")

# test
df = pd.read_csv('data/mnist_test.csv')
npa = df.to_numpy()
correct = 0
print("testing")
for im in npa:
    correct+=int(get_pred(net.compute((1/256)*np.array([list(im[1::])]).transpose()))==im[0])
# save to text file
wstr = net.dump_network_to_string()
with open(f"tt-mnist-classification-weights{time.time()}","w") as f:
    f.write(wstr)
# currently 90%
print(f"% correct on test data: {100*correct/len(npa)}")