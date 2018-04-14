import layer
import functions as f
import numpy as np

class network:
    def __init__(self):
        self.layers = []

    def layer(self, num_nodes, function, input_num = None):
        errvalue = False
        if (input_num == None and self.layers != []):
                innum = (self.layers[len(self.layers)-1]).nnodes
        elif (self.layers != [] and  input_num != (self.layers[len(self.layers)-1]).nnodes):
            print ("ERR: Input dimension mismatch, layer not created.")
            errvalue = True
        elif (self.layers == [] and input_num == None):
            print ("ERR: No input dimension specified.")
            errvalue = True
        elif (self.layers == [] and input_num != None):
                innum = input_num
        if (errvalue == False):
            self.layers += [layer.layer(innum, num_nodes, function)]
        return errvalue

    def setweights(self, layer, weights, node = None):
        self.layers[layer].setweights(weights, node)
        return True

    def setbias(self, layer, bias, node = None):
        self.layers[layer].setbias(bias, node)
        return True

    def forward(self, inp, keep_vals = False):
        output = [inp]
        count = 0

        while (len(self.layers) != count):
            output += [(self.layers[count]).compute(output[count])]
            count += 1
        if (keep_vals == False):
            return output[count]
        else:
            return output

    def backprop(self, inp, expected, eta):
        out = self.forward(inp, True)
        depth = len(self.layers)
        updated = [[]] * depth
        delta = [[]] * depth
        # Compute deltas, and populate updated with old weights
        delta [depth-1] = np.array(np.multiply(np.subtract(np.array(expected), np.array(out[depth])), eval('f.' + ((self.layers[depth-1]).func).__name__ + '_der')(np.array(out[depth]))))
        for i in range(depth):
            out[i] = np.append(out[i], 1)

        for i in range(depth - 2, -1, -1):
            fder = eval('f.' + ((self.layers[i]).func).__name__ + '_der')
            weights = np.array(self.layers[i+1].weights)
#            print weights, delta[i+1]
            delta[i] = np.array(np.multiply(np.dot(np.transpose(np.array(weights)), delta[i+1]), fder(out[i+1])))
            delta[i] = delta[i][0:len(delta[i])-1]
            updated[i+1] = weights
            
        updated[0] = np.array(self.layers[0].weights)
        
#        print 'DELTA: ', delta, 'UPDATED: ', updated, 'OUT: ', out

        # Compute new weights

        for i in range(depth):
            updated[i] += eta * np.dot(delta[i].reshape(-1,1), out[i].reshape(1, -1))
        #    updated[i][len(updated[i])-1] -= eta * delta[i]
        for i in range(depth):
            self.setweights(i, updated[i])

    def save(self, savefile):
        accum = []
        for i in self.layers:
            accum = accum + [i.weights]
        np.save(savefile, np.array(accum))

    def load(self, savefile):
        weights = np.load(savefile)
        for i in range(len(self.layers)):
            self.layers[i].setweights(weights[i])
