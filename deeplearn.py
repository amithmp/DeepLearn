"""
@author: Amith_Parameshwara
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.datasets

# =============================================================================
#          Deep learning functions for classification and regression
# =============================================================================
#  Activations supportd: relu, leakyrelu, tanh, sigmoid, softmax
#  Regularizations supported: dropout, L2
#  Optimizers supported: gradient descent, momentum, adam
#  Cost functions supported: binary cross entropy cost (binary classification),
#                            categorical cross entropy (multiclass classification),
#                            mse and rmse (regression)
#  Other features supported: mini batch, learning rate decay 
#  Weight initialization: he
#  Data sets are expected to be in the format (features,examples)
# =============================================================================

# =============================================================================
#   Example 1
#       n_layer_learn().demo()
# =============================================================================
#   Example 2
#       import deeplearn
#       X_train, Y_train, X_test, Y_test = <Your data sets>
#       d_model = deeplearn.n_layer_learn()
#       d_model.add_layers(X_train.shape[0],"none")
#       d_model.add_layers(100,"relu",lambd=0.01,dropout_ratio=0.2)
#       d_model.add_layers(50,"leakyrelu",lambd=0.01,leaky_alpha=-0.05,dropout_ratio=0.1)
#       d_model.add_layers(30,"relu",lambd=0.01,dropout_ratio=0.1)
#       d_model.add_layers(20,"relu",lambd=0.1,dropout_ratio=0.1)
#       d_model.add_layers(Y_train.shape[0],"sigmoid")
#       d_model.train(X_train, Y_train, optimizer="adam",epochs = 400,learning_rate = 0.03, learning_rate_decay=0.003, momentum_beta = 0.9, rmsprop_beta=0.99, batch_size=16, verbose=True, debug=False)
#       predicted_values = d_model.predict(X_test)
#       deepnet_test_accuracy,_,_,_ = d_model.deepnet_evaluate(X_test, Y_test)
# =============================================================================

# =============================================================================
#   Known Issues
#       Unstable for higher values of learning rate in case of ADAM and multi-class classification using softmax
#       Remedy - try with diffirent/smaller values of learning rate as well as learning rate decay
# 
# =============================================================================


class n_layer_learn:
    # Initialize all variables in the class
    def __init__(self):
        np.random.seed(1805)
        self.layers_dims = []
        self.activations = []
        self.leaky_alpha = []
        self.lambd = []
        self.learning_rate = 0
        self.debug = False
        self.verbose = False
        self.num_layers = 0
        self.parameters = {}
        self.momentum = 0
        self.optimizer = "gd"
        self.adam_epsilon = 1e-08
        self.rmsprop_beta = 0
        self.momentum_beta = 0
        self.batch_size = 0
        self.learning_rate_delay = 0
        self.dropout_ratio = []
        self.caches = []
        self.cost_function = 'none'
        
    # add a new layer to the model. Inputs are number of neurons, activation method, regularization lambda, dropout ratio, and alpha for leaky relu activation
    def add_layers(self,layers_dim, activation, lambd= 0, dropout_ratio=0, leaky_alpha=-1000):
        self.layers_dims.append(layers_dim)
        self.activations.append(activation)
        self.lambd.append(lambd)
        if(dropout_ratio == 0):
            self.dropout_ratio.append(1)
        else:
            self.dropout_ratio.append(1-dropout_ratio)
        if(activation=="leakyrelu"):
            assert(leaky_alpha != -1000)
        self.leaky_alpha.append(leaky_alpha)
        if(self.num_layers > 0):
            self.initialize_parameters(self.num_layers)
        self.num_layers = self.num_layers + 1


    # Prints the structures of the neural network
    def print_structure(self):
        for i in range(0,self.num_layers):
            print("Layer %i: %i neurons with activation %s" %(i,self.layers_dims[i],self.activations[i]))
          
        
    def leakyrelu(self, Z, alpha=-0.1):
        if(self.debug):
            print("Entering leakyrelu")
        A = np.maximum(alpha,Z)
        assert(A.shape == Z.shape)
        cache = Z 
        return A, cache
    
    
    def tanh(self, Z):
        if(self.debug):
            print("Entering tanh")
        cache = Z
        A = (1-np.exp(-2*Z)) / (1+np.exp(-2*Z))    
        return A, cache
    
    
    def sigmoid(self, Z):
        if(self.debug):
            print("Entering sigmoid") 
        A = np.exp(-np.logaddexp(0, -Z))
        cache = Z
        return A, cache
    
    def relu(self,Z):
        if(self.debug):
            print("Entering relu")
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        cache = Z 
        return A, cache
    
    
    def softmax(self, Z):
        if(self.debug):
            print("Entering softmax")
        cache = Z
        Zexp = np.exp(np.logaddexp(0, Z))
        Zsum = np.sum(Zexp,axis=0,keepdims=True)
        A = Zexp / Zsum
        return A, cache
    
    # Calculates derivative of relu activation using the formula dZ = dA * d(relu) 
    def relu_backward(self, dA, cache):
        if(self.debug):
           print("Entering relu_backward")
        Z = cache
        dZ = np.array(dA, copy=True) 
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ
    
    # Calculates derivative of sigmoid activation using the formula dZ = dA * d(sigmoid)  
    def sigmoid_backward(self, dA, cache):
        if(self.debug):
            print("Entering sigmoid_backward")
        Z = cache
        s,_ = self.sigmoid(Z)
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ  
    
    # Calculates derivative of leakyrelu activation using the formula dZ = dA * d(leakyrelu) 
    def leakyrelu_backward(self, dA, cache, leaky_alpha):
        if(self.debug):
            print("Entering leakyrelu_backward")
        Z = cache
        dZ = np.array(dA, copy=True) 
        dZ[Z <= leaky_alpha] = 0
        assert (dZ.shape == Z.shape)
        return dZ
        
    # Calculates derivative of softmax activation using the formula dZ = dA * d(softmax) 
    def softmax_backward(self, dA, cache):
        if(self.debug):
            print("Entering softmax_backward")
        Z = cache
        s,_ = self.softmax(Z)
        dZ = s* (1-s)
        dZ = dZ * dA
        assert (dZ.shape == Z.shape)
        return dZ
    
    
    # Calculates derivative of tanh activation using the formula dZ = dA * d(tanh) 
    def tanh_backward(self, dA, cache):
        if(self.debug):
            print("Entering tanh_backward")
        Z = cache
        s = (1-np.exp(-2*Z)) / (1+np.exp(-2*Z))  
        dZ = dA * (1-np.square(s))
        assert (dZ.shape == Z.shape)
        return dZ  
    
    
    # Initiatives weight and bias vectors 
    def initialize_parameters(self,layer_num):
          if(self.debug):
              print("Entering initialize_parameters for layer %i" %layer_num)
            
          np.random.seed(3)
    
          self.parameters['W' + str(layer_num)] = np.random.randn(self.layers_dims[layer_num],self.layers_dims[layer_num-1])*np.sqrt(2./self.layers_dims[layer_num-1])
          self.parameters['b' + str(layer_num)] = np.zeros((self.layers_dims[layer_num],1))
           
          assert(self.parameters['W' + str(layer_num)].shape == (self.layers_dims[layer_num], self.layers_dims[layer_num-1]))
          assert(self.parameters['b' + str(layer_num)].shape == (self.layers_dims[layer_num], 1))
                
          
    # Initializes velocity vector for momentum optimizer
    def initialize_momentum(self):
        if(self.debug):
              print("Entering initialize_parameters_momentum")     
        L = self.num_layers - 1
        v = {}
        for l in range(L):
         v["dW" + str(l+1)] = np.zeros((self.parameters['W' + str(l+1)].shape[0],self.parameters['W' + str(l+1)].shape[1]))
         v["db" + str(l+1)] = np.zeros((self.parameters['b' + str(l+1)].shape[0],1))
        return v
    
    
    # Initializes velocity and RMSprop vector for adam optimizer
    def initialize_adam(self):
        if(self.debug):
              print("Entering initialize_parameters_adam")       
        L = self.num_layers - 1
        v = {}
        s = {}
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros((self.parameters['W' + str(l+1)].shape[0],self.parameters['W' + str(l+1)].shape[1]))
            v["db" + str(l+1)] = np.zeros((self.parameters['b' + str(l+1)].shape[0],1))
            s["dW" + str(l+1)] = np.zeros((self.parameters['W' + str(l+1)].shape[0],self.parameters['W' + str(l+1)].shape[1]))
            s["db" + str(l+1)] = np.zeros((self.parameters['b' + str(l+1)].shape[0],1))
        return v,s
    
    # Calculate linear Z as Z = W*A + b
    def linear_forward(self,A, W, b):
        if(self.debug):
            print("Entering linear_forward")
        Z = np.dot(W,A) + b   
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache
    
    # Calculate the activation for the layer
    def linear_activation_forward(self,A_prev, W, b, activation, leaky_alpha):
        if(self.debug):
            print("Entering linear_activation_forward with activation %s" %(activation))
        Z, linear_cache = self.linear_forward(A_prev,W,b)
        if activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = self.relu(Z)
        elif activation =='leakyrelu':
            A, activation_cache = self.leakyrelu(Z,leaky_alpha)
        elif activation =='tanh':
            A, activation_cache = self.tanh(Z)
        elif activation =='softmax':
            A, activation_cache = self.softmax(Z)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
    
        return A, cache
    
    
    # Move forward over all layers, calculating linear Zs and activations
    def L_model_forward(self,X):
        
        if(self.debug):
            print("Entering L_model_forward")
            
        caches = []
        A = X
        L = self.num_layers-1       

        for l in range(1, L):
            A_prev = A 
            if(self.debug):
                print("forward prop in layer %s with activation %s, updating W%i" %(l,self.activations[l],l))
            A, cache = self.linear_activation_forward(A_prev, self.parameters["W"+str(l)], self.parameters["b"+str(l)], self.activations[l], self.leaky_alpha[l])
            self.parameters["D"+str(l)] = np.random.rand(A.shape[0],A.shape[1])
            self.parameters["D"+str(l)] = (self.parameters["D"+str(l)] < self.dropout_ratio[l])
            A = np.multiply(A,self.parameters["D"+str(l)])
            A = np.divide(A,self.dropout_ratio[l])
            caches.append(cache)

        if(self.debug):
            print("forward prop in layer %i with activation %s, updating W%i" %(L,self.activations[L],L))
        AL, cache = self.linear_activation_forward(A, self.parameters["W"+str(L)], self.parameters["b"+str(L)], self.activations[L], self.leaky_alpha[L])
        self.parameters["D"+str(L)] = np.zeros((AL.shape[0],AL.shape[1]))
        self.parameters["D"+str(L)] = self.parameters["D"+str(L)]+1
        caches.append(cache)
               
        assert(AL.shape == (self.layers_dims[self.num_layers-1],X.shape[1]))
        return AL, caches
    
    
    # Compute the cross entropy cost for binary or multi class 
    def compute_cost(self,AL, Y):
        
        if(self.debug):
            print("compute_cost")
        m = Y.shape[1]
    
        cost_regularization = 0
        AL=np.where(AL==1,0.999999,AL)
        AL=np.where(AL==0,0.000001,AL)
        
        if(self.cost_function == "categorical_cross_entropy"):
            cost_main = -1./m*np.nansum(np.multiply(Y,np.log(AL)))
        elif(self.cost_function == "binary_cross_entropy"):
            cost_main = -1./m*(np.nansum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y,np.log(1-AL))))
        elif(self.cost_function == "rmse"):
            cost_main = np.sqrt(np.sum(np.square(AL-Y))/(AL.shape[0]*AL.shape[1]))
        elif(self.cost_function == "mse"):
            cost_main = (np.sum(np.square(AL-Y))/(AL.shape[0]*AL.shape[1]))
        else:
            assert(False)
            
        for l in range (1,self.num_layers):
            cost_regularization =  cost_regularization + np.sum(np.square(self.parameters["W"+str(l)])) 
        cost_regularization =  1/m * self.lambd[l-1] / 2 * cost_regularization
        cost_main = np.squeeze(cost_main)     
        cost = cost_main + cost_regularization
        assert(cost.shape == ())
        
        if(self.debug):
            print ("Computed Main Cost %f" %(cost_main))
            print ("Computed Regularization Cost %f" %(cost_regularization))
 
        return float(cost)
    
    
    # Compute derivatives for weight and bias for a layer
    def linear_backward(self, dZ, cache, lambd):
       
        if(self.debug):
            print("Entering linear_backward")
        A_prev, W, b = cache
        m = A_prev.shape[1]
    
        dW = 1/m * np.dot(dZ,A_prev.T) + ((lambd/m)*W)
        db = 1/m * np.sum(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db
    
    
    # Compute derivative of activation functions in a layer
    def linear_activation_backward(self, dA, cache, lambd, activation, leaky_alpha):
       
        if(self.debug):
            print("Entering linear_model_backward with activation %s" %activation)
            
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = self.relu_backward(dA,activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA,activation_cache)
        elif activation == "leakyrelu":
            dZ = self.leakyrelu_backward(dA,activation_cache,leaky_alpha)
        elif activation == "tanh":
            dZ = self.tanh_backward(dA,activation_cache)
        elif activation == "softmax":
            dZ = self.softmax_backward(dA,activation_cache)
        
        dA_prev, dW, db = self.linear_backward(dZ,linear_cache,lambd)
        
        return dA_prev, dW, db
    
    
    # Move backward over all layers calculating derivatives for activations, weight and bias vectors
    def L_model_backward(self, AL, Y, caches):
      
        grads = {}
        L = self.num_layers - 1 
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) 
        AL=np.where(AL==1,0.999999,AL)
        AL=np.where(AL==0,0.000001,AL)
        
        if(self.cost_function == "categorical_cross_entropy"):
            dAL = AL-Y
        elif(self.cost_function == "binary_cross_entropy"):
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        elif(self.cost_function == "rmse"):
            x = AL-Y
            x=np.where(x==0,0.0000001,x)
            dAL = x / (np.sqrt(m)*np.abs(x))
        elif(self.cost_function == "mse"):
            x = AL-Y
            dAL = 2*x/m
        else:
            assert(False)
            
        current_cache = caches[L-1]

        if(self.debug):
            print("backward prop in layer %s with activation %s, updating dW%i" %(L,self.activations[L],L))
        dA_L_temp, grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, self.lambd[L], self.activations[L],self.leaky_alpha[L])
        dA_L_temp = dA_L_temp * self.parameters["D"+str(L-1)]
        dA_L_temp = dA_L_temp / self.dropout_ratio[L]
        grads["dA" + str(L)] = dA_L_temp
        
        for l in reversed(range(L-1)):
            if(self.debug):
                print("backward prop in layer %s with activation %s, updating dW%i" %(l+1,self.activations[l+1],l+1))
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache, self.lambd[l+1], self.activations[l+1],self.leaky_alpha[l+1])
            if(l != 0):
                dA_prev_temp = dA_prev_temp * self.parameters["D"+str(l)]
                dA_prev_temp = dA_prev_temp / self.dropout_ratio[l+1]
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
            
        return grads
    
    
    # Update weight and bias vectors using derivatives for Gradient descent optimizer
    def update_parameters_gd(self,grads):
        
        if(self.debug):
            print("Entering update parameters")
        L = self.num_layers - 1 # number of layers in the neural network
    
        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - self.learning_rate*grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate*grads["db" + str(l+1)]
            
            #print(np.max(self.parameters["W" + str(l+1)]))


    # Update weight and bias vectors using derivatives for momentum optimizer 
    def update_parameters_momentum(self,grads,v):
  
        if(self.debug):
            print("Entering update parameters momentum")
            
        L = self.num_layers - 1
    
        for l in range(L):  
            v["dW" + str(l+1)] = self.momentum_beta * v["dW" + str(l+1)] + (1-self.momentum_beta)* grads["dW" + str(l+1)]
            v["db" + str(l+1)] = self.momentum_beta * v["db" + str(l+1)] + (1-self.momentum_beta)* grads["db" + str(l+1)]
          
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - self.learning_rate * v["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate * v["db" + str(l+1)]
            
           # print(np.max(self.parameters["W" + str(l+1)]))

        return v
    
    
    # Update weight and bias vectors using derivatives for adam optimizer
    def update_parameters_adam(self,grads, v, s, t):
        
        if(self.debug):
            print("Entering update parameters adam")
            
        L = self.num_layers - 1               
        v_corrected = {}                         
        s_corrected = {}                        
    
        for l in range(L):
            v["dW" + str(l+1)] = self.momentum_beta * v["dW" + str(l+1)] + (1-self.momentum_beta)* grads["dW" + str(l+1)]
            v["db" + str(l+1)] = self.momentum_beta * v["db" + str(l+1)] + (1-self.momentum_beta)* grads["db" + str(l+1)]

            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-np.power(self.momentum_beta,t))
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-np.power(self.momentum_beta,t))

            s["dW" + str(l+1)] = self.rmsprop_beta * s["dW" + str(l+1)] + (1-self.rmsprop_beta)* np.square(grads["dW" + str(l+1)])
            s["db" + str(l+1)] = self.rmsprop_beta * s["db" + str(l+1)] + (1-self.rmsprop_beta)* np.square(grads["db" + str(l+1)])
            
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-np.power(self.rmsprop_beta,t))
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-np.power(self.rmsprop_beta,t))
      
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - (self.learning_rate * v_corrected["dW" + str(l+1)] / np.sqrt(s_corrected["dW" + str(l+1)]+self.adam_epsilon))
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - (self.learning_rate * v_corrected["db" + str(l+1)] / np.sqrt(s_corrected["db" + str(l+1)]+self.adam_epsilon))

          #  print(np.max(self.parameters["W" + str(l+1)]))
        
        return v, s


    # Create random mini batches using the given batch size
    def random_mini_batches(self,X, Y, seed):
        
        np.random.seed(seed)
        m = X.shape[1] 
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        num_minibatches = math.floor(m/self.batch_size)
        for k in range(0, num_minibatches):
            mini_batch_X = shuffled_X[:, self.batch_size*k:self.batch_size*(k+1)]
            mini_batch_Y = shuffled_Y[:, self.batch_size*k:self.batch_size*(k+1)]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        if m % self.batch_size != 0:
            mini_batch_X = shuffled_X[:, (self.batch_size*num_minibatches):m]
            mini_batch_Y = shuffled_Y[:, (self.batch_size*num_minibatches):m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        if(self.debug):
            print("Created %i mini batches" %(num_minibatches))
    
        return mini_batches
    
    
    # train the model over given number of epochs, iterating over steps of forward prop, 
    # cost calculation, back propagation, parameter updation 
    def train(self, X, Y, optimizer, cost_function = "none", learning_rate = 0.0075, epochs = 3000, batch_size=-1, momentum_beta=0.9,rmsprop_beta=0.999, learning_rate_decay=0, adam_epsilon=1e-8, verbose=False, debug=False, print_iter=100):
       
        np.random.seed(1)
        costs = []                         
        grads = {}
        t = 0
        L = self.num_layers - 1
        seed = 23
        
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum_beta = momentum_beta
        self.rmpsprop_beta = rmsprop_beta
        self.debug = debug
        self.verbose = verbose
        self.learning_rate_decay = learning_rate_decay
        self.adam_epsilon = adam_epsilon
        self.cost_function = cost_function

        if(self.cost_function == "none"):
            if(len(np.unique(Y)) == 2):
                if(self.layers_dims[self.num_layers-1] > 1):
                    self.cost_function = "categorical_cross_entropy"
                else:
                    self.cost_function = "binary_cross_entropy"
            else:
                self.cost_function = "rmse"

        assert(self.layers_dims[0] == X.shape[0])
        assert(self.layers_dims[self.num_layers - 1] == Y.shape[0])        

        if(batch_size==-1):
           self.batch_size = X.shape[1] 
        else:
            self.batch_size = batch_size
        
        if optimizer == "gd":
            pass
        elif optimizer == "momentum":
            v = self.initialize_momentum()
        elif optimizer == "adam":
            v,s = self.initialize_adam()
            
        for i in range(epochs):
            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, seed)
            minibatch_num = -1
            self.learning_rate = self.learning_rate / (1+self.learning_rate_decay*i)
            
            for minibatch in minibatches:
                minibatch_num = minibatch_num + 1
                if(self.debug):
                    print("Entering epoch %i processing mini batch %i" %(i,minibatch_num))
                (minibatch_X, minibatch_Y) = minibatch
                
                AL, caches = self.L_model_forward(minibatch_X)
                cost = self.compute_cost(AL, minibatch_Y)
                grads = self.L_model_backward(AL, minibatch_Y, caches)
                if self.optimizer == "gd":
                    self.update_parameters_gd(grads)
                elif self.optimizer == "momentum":
                    v = self.update_parameters_momentum(grads,v)
                elif self.optimizer == "adam":
                    t = t+1
                    v,s = self.update_parameters_adam(grads,v,s,t)
                else:
                    assert(False)

            if self.verbose and i % print_iter == 0:
                print ("Cost after epoch %i: %f" %(i, cost))
                costs.append(cost)
                if(self.cost_function == 'rmse' or self.cost_function == 'mse'):
                     Y_prediction = self.predict(X)
                     train_accuracy = self.compute_cost(Y_prediction, Y)
                     print(self.cost_function," after epoch",i,":{0:.2f} "  .format(train_accuracy))
                else:
                     train_accuracy,_,_,_,_ = self.evaluate(X,Y)
                     print("Train accuracy after epoch",i,":{0:.2f} %"  .format(train_accuracy))
    
        if self.verbose:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
        
    # Predicts the outcome for new values of X
    def predict(self, X, threshold=0.5):
        A2, cache = self.L_model_forward(X)
        predictions = np.zeros((A2.shape[0],A2.shape[1]))
        if(self.activations[self.num_layers-1] == "softmax"):
            pos = np.argmax(A2, axis=0)                
            for i in range(0,A2.shape[1]):
                predictions[pos[i],i] = 1
        elif(self.activations[self.num_layers-1] == "sigmoid"):
            predictions = np.where(A2>=threshold, 1, 0)
        else:
            predictions = A2
        return predictions
    
    
    # for Binary classification, returns True Positives, False Positives, True Negatives and 
    # False negatives. For multi-class classification, returns % of correct classiciations 
    def measure_classification(self, Y_values, P_values):
        tp,fp,tn,fn = 0,0,0,0
        accuracy = 0
        if(len(np.unique(Y_values)) > 2):
            for i in range(0,len(P_values)):
                if(Y_values[i] == P_values[i]):
                    accuracy += 1
            return (accuracy * 100) / len(P_values),0,0,0
        else:
            for i in range(0,len(P_values)):
                if(Y_values[i] == 1):
                    if(P_values[i] == 1):
                        tp += 1
                    else:
                        fn += 1
                else:
                    if(P_values[i] == 1):
                        fp += 1
                    else:
                        tn += 1
            return tp,fp,tn,fn
       
    
    # Predicts and evaluates performance for new values of x.
    # For regression, returns RMSE or MSE as per the cost function
    # For classification, returns accuracy,tp,fp,tn,fn for binary classification and % of correct 
    # classifications for multi-class classifications
    def evaluate(self, X_test, Y_test, threshold=0.5):
        Y_prediction_test = self.predict(X_test,threshold)
        if(self.cost_function == 'rmse' or self.cost_function == 'mse'):
            test_accuracy = self.compute_cost(Y_prediction_test, Y_test)
            return test_accuracy,0,0,0
        elif(self.activations[self.num_layers-1] == "softmax"):
            Y_values = Y_test.argmax(0)
            P_values = Y_prediction_test.argmax(0)
            test_accuracy,_,_,_ = self.measure_classification(Y_values,P_values)
            return test_accuracy,0,0,0,0
        else:
            Y_values = Y_test.reshape(Y_test.shape[1],1).squeeze()
            P_values = Y_prediction_test.reshape(Y_prediction_test.shape[1],1).squeeze()
            tp,fp,tn,fn = self.measure_classification(Y_values,P_values)
            test_accuracy = ((tp+tn) / len(Y_values))*100
            return round(test_accuracy,1),round(tp*100/Y_test.shape[1],1),round(fp*100/Y_test.shape[1],1),round(tn*100/Y_test.shape[1],1),round(fn*100/Y_test.shape[1],1)

    # Demo of the deep learning model 
    def demo(self):
        X_train, Y_train = sklearn.datasets.make_circles(n_samples=500, noise=.04)
        X_test, Y_test = sklearn.datasets.make_circles(n_samples=200, noise=.04)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=40, cmap=plt.cm.Spectral);
        plt.show()
        X_train = X_train.T
        Y_train = Y_train.reshape((1, Y_train.shape[0]))
        X_test = X_test.T
        Y_test = Y_test.reshape((1, Y_test.shape[0]))
        self.add_layers(X_train.shape[0],"none")
        self.add_layers(100,"relu",lambd=0.01,dropout_ratio=0.1)
        self.add_layers(30,"relu",lambd=0.01,dropout_ratio=0.1)
        self.add_layers(20,"relu",lambd=0.1,dropout_ratio=0)
        self.add_layers(Y_train.shape[0],"sigmoid")
        self.train(X_train, Y_train, optimizer="gd",epochs = 600,learning_rate = 0.03, learning_rate_decay=0, momentum_beta = 0.9, rmsprop_beta=0.99, batch_size=16, adam_epsilon=1e-8, verbose=True, debug=False)
        self.print_structure()
        deepnet_train_accuracy,_,_,_ = self.evaluate(X_train, Y_train)
        deepnet_test_accuracy,_,_,_ = self.evaluate(X_test, Y_test)
        print("Train accuracy: {0:.2f} %".format(deepnet_train_accuracy))
        print("Test accuracy: {0:.2f} %".format(deepnet_test_accuracy))
