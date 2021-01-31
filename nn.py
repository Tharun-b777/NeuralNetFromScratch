
import matplotlib as plt
import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

np.random.seed(0)
inputs=[[1.2,5.2,6.7,5],[2,3,5,6],[4,0.3,6,7]]
weights1 = [[3, 12, 1,7], [3.3, 2.7, 1.9,5], [3.2, 1.90, 0.7,2]]
weights2=[[-0.1,0.14,0.5],[0.9,7.1,0.9],[1.2,-4.1,5.2]]
biases1=[3,4,0.5]
biases2=[0.1,3,5]
layer1_output = np.dot(inputs, np.array(weights1).T)+biases1
layer2_output=np.dot(layer1_output,weights2)+biases2
#print(layer1_output)
#print(layer2_output) 
 
X = [[1.2, 5.2, 6.7, 5], 
     [2, 3, 5, 6], 
     [4, 0.3, 6, 7]]

#Auto initalisinf weights and biases
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
                                       #rows     #colomns     
        self.weights=0.1*np.random.rand(n_inputs,n_neurons) 
        self.biases=np.ones((1,n_neurons))
        
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights) + self.biases  
    def backprop(self,inputs):
        pass


layer1=Layer_Dense(4,5)
layer2=Layer_Dense(5,4)
layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


# inputs = [0, 2, -1, 3.3, 2.7, 1.1, 2, 2]
#Rectified linear function
X,y =spiral_data(100,3)

class Activation_ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)


                      #inputs #neurons
layer_spiral=Layer_Dense(2,5)
activation1=Activation_ReLU()
layer_spiral.forward(X)
activation1.forward(X)
print(activation1.output)



def backprop(z,y):
    del=2*(z-y)*
    