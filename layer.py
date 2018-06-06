# Understood this stuff from "Neural Networks and Deep Learning" by Michael Nielsen (http://neuralnetworksanddeeplearning.com/) 
# Improved version from Neural-Net supersedes node.py and layer.py
import numpy as np #www.numpy.org

class layer:
    def __init__(self, in_num, n_nodes, func):
        self.input = in_num
        self.func = func
        self.nnodes = n_nodes

        self.weights = ((2 * np.random.random_sample((self.nnodes, self.input + 1))) - 1) 

    def compute(self, inp):
        inp_bi = np.append(inp, 1)
        return self.func(np.dot(self.weights, inp_bi))

    def setweights(self, inlist, node = None):
        if (node == None):
            self.weights = np.array(inlist)
        else:
            self.weights[node] = np.array(inlist)

    def setbias(self, inlist, node = None):
        if (node == None):
            for i in range(self.nnodes):
                self.weights[i][self.input] = inlist
        else:
            self.weights[node][len(self.input)] = inlist
