#------------------------------------------------------------
# A neural net class that fits a net (with back propagation)
#-----------------------------------------------------------
import random
import numpy as np

class NeuralNet(object):

  """ Constructor
  #   layers: vector of length numLayers containing the
  #          the number of neurons in each layer
  #           e.g., layers = (4, 3, 2) -> 4 input values, 3 neurons in hidden layer, 2 output values
  #   biases: initialized with random numbers except for first layer
  #           e.g., biases = [ [b11, b21, b31]^T, 
  #                            [b12, b22]^T ]
  #   weights: initialized with random numbers 
  #           e.g., [ [[w111, w121, w131, w141], 
  #                    [w211, w221, w231, w241],
  #                    [w311, w321, w331, w341]],
  #                   [[w112, w122, w132],
  #                    [w212, w222, w232]] ]
  """
  def __init__(self, layers):

    self.numLayers = len(layers)
    self.numNeurons = layers
    self.biases = [np.random.randn(layer, 1) for layer in layers[1:]]
    self.weights = [np.random.randn(layer2, layer1) 
                    for layer1, layer2 in zip(layers[:-1], layers[1:])]
    

  """ Batch stochastic gradient descent to find minimum of objective function
  #   training_data:  [(x1,y1),(x2,y2),....]
  #                   where x1, x2, x3, ... are input data vectors
  #                         y1, y2, y3, ... are labels
  #   max_iterations: number of iterations
  #   batch_size:     size of training batch
  #   learning_rate:  gradient descent parameter 
  """
  def batchStochasticGradientDescent(self, training_data, max_iterations, batch_size,
                                     learning_rate):

    # Get the number of training images
    nTrain = len(training_data)

    # Loop thru iterations
    for it in xrange(max_iterations):

      # Shuffle the training data
      np.random.shuffle(training_data)

      # Choose subsets of the training data
      batches = [ training_data[start:start+batch_size]
                  for start in xrange(0, nTrain, batch_size) ]

      # Loop thru subsets
      for batch in batches:
        self.updateBatch(batch, learning_rate)
      
      #print "Iteration {0} complete".format(it)
        
    #print "weights = ", self.weights
    #print "biases = ", self.biases

  """ Partial update of weights and biases using gradient descent
  #   with back propagation
  """
  def updateBatch(self, batch, learning_rate): 

    # Initialize gradC_w and gradC_b
    gradC_w = [np.zeros(w.shape) for w in self.weights]
    gradC_b = [np.zeros(b.shape) for b in self.biases]
    
    # Loop through samples in the batch
    for xx, yy in batch:
    
      # Compute correction to weights & biases using forward and backprop
      delta_gradC_w, delta_gradC_b = self.updateGradient(xx, yy)

      # Update the gradients
      gradC_w = [grad + delta_grad for grad, delta_grad in zip(gradC_w, delta_gradC_w)]
      gradC_b = [grad + delta_grad for grad, delta_grad in zip(gradC_b, delta_gradC_b)]

    # Update the weight and biases
    self.weights = [ weight - (learning_rate/len(batch))*grad
                     for weight, grad in zip(self.weights, gradC_w) ]
    self.biases  = [ bias - (learning_rate/len(batch))*grad
                     for bias, grad in zip(self.biases, gradC_b) ]

  # Forward and then backpropagation to compute the gradient of the objective function
  def updateGradient(self, xx, yy):

    # Reshape into column vectors
    xx = np.reshape(xx, (len(xx), 1))
    yy = np.reshape(yy, (len(yy), 1))
    
    # Initialize gradC_w and gradC_b
    gradC_w = [np.zeros(w.shape) for w in self.weights]
    gradC_b = [np.zeros(b.shape) for b in self.biases]

    # Compute forward pass through net
    # Initial activation value = input value
    activationValue = xx
    activationValues = [xx]
    layerOutputValues = []
    
    # Loop through layers
    for weight, bias in zip(self.weights, self.biases):

      #print weight.shape
      #print activationValue.shape
      layerOutputValue = np.dot(weight, activationValue) + bias
      layerOutputValues.append(layerOutputValue)

      activationValue = self.activationFunction(layerOutputValue)
      activationValues.append(activationValue)
      
    # Compute backpropagation corrections
    # Initial deltas
    delta = self.derivOfCostFunction(activationValues[-1], yy) * self.derivActivationFunction(layerOutputValues[-1])

    gradC_b[-1] = delta
    gradC_w[-1] = np.dot(delta, activationValues[-2].transpose())

    # Loop backward thru layers
    for layer in xrange(2, self.numLayers):

      layerOutputValue = layerOutputValues[-layer]
      derivActivation = self.derivActivationFunction(layerOutputValue)
  
      delta = np.dot(self.weights[-layer + 1].transpose(), delta)*derivActivation

      gradC_b[-layer] = delta
      gradC_w[-layer] = np.dot(delta, activationValues[-layer-1].transpose())
    
    # Return updated gradients
    return(gradC_w, gradC_b)

  # The activation function
  def activationFunction(self, xx):

    return 1.0/(1.0 + np.exp(-xx))

  # Derivative of activation function
  def derivActivationFunction(self, xx):

    return self.activationFunction(xx)*(1.0 - self.activationFunction(xx))
  
  # Derivative of the cost function with respect to output values
  def derivOfCostFunction(self, xx, yy):
    return (xx - yy)

  # The feedforward output computation for the network
  #   inputVector: (n, 1) array
  #                n = number of inputs to network
  #   outputVector: ????
  def forwardCompute(self, inputVector):

    for bias, weight in zip(self.biases, self.weights):
      xx = np.dot(weight, inputVector) + bias
      inputVector = self.activationFunction(xx)

    return inputVector
  
