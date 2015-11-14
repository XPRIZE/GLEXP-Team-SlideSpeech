#
# Fit a neural net model that can predict labelled letters based on images
# saved by lettersketch
#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import Image
import codecs, json

# Import the Neutral net class
from NeuralNetUtils import NeuralNet

# # <font color='green'>Load the data needed to fit the neural net</font>
# ### Get the names of the files and letters saved by the lettersketch app
dirName = "../lettersketch/assets/train_images/UpperCase/StraightLines/"
fileNames = []
fileLetters = []
for fileName in os.listdir(dirName):
    if fileName.endswith(".png") and (not "__" in fileName):
       fileNames.append(dirName+fileName)
       letter = fileName.split("_")[1]
       fileLetters.append(letter)
    
#print fileNames

# ### Read in the letters saved by the lettersketch app after converting 
# ### into grayscale and resizing
data = np.array([np.asarray(Image.open(fileName).convert('L').resize((28, 28), Image.NEAREST)) 
                 for fileName in fileNames], dtype = np.float64)
print data.shape

# ### Reshape the image arrays
data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
print data.shape

# ### Convert white into black background
for dt in np.nditer(data, op_flags=['readwrite']):
    dt[...] = dt/255.0
    if (dt == 1.0):
        dt[...] = 0.0

# ### Load the test data, Convert white into black background, 
# ### Reshape the data
testDirName = "../lettersketch/assets/test_images/"
testNames = []
for fileName in os.listdir(testDirName):
    if fileName.endswith(".png") and ("__" in fileName):
       testNames.append(testDirName+fileName)

testData = np.array([np.asarray(Image.open(fileName).convert('L').resize((28, 28), Image.NEAREST)) 
                 for fileName in testNames], dtype = np.float64)
print testData.shape

testData = testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2])
print testData.shape

for dt in np.nditer(testData, op_flags=['readwrite']):
    dt[...] = dt/255.0
    if (dt == 1.0):
        dt[...] = 0.0

# Plot samples
imgplot = plt.imshow(data[9].reshape(28,28), cmap="gray")
testplot = plt.imshow(testData[1].reshape(28,28), cmap ="gray")

# ### Create a label dictionary for the letters in the training set
labelDict = {'E':0, 'F':1, 'H':2, 'I':3, 'L':4, 'T':5}
print fileLetters

# ### Assign labels to the images in the training set
fileLabels = [labelDict[letter] for letter in fileLetters]
print fileLabels

# ### Vectorize the labels
def vectorizeLabels(label):
    vector = np.zeros((6))
    vector[label] = 1.0
    return vector

dataLabels = np.array([vectorizeLabels(label) for label in fileLabels])
print dataLabels[0]

# ### Join data and data labels
print data.shape
print dataLabels.shape
training_data = zip(data, dataLabels)
#print training_data[0]

# # <font color='green'>Fit the neural net to the input data</font>
# In[95]:
testNN = NeuralNet([28*28, 49, 16, 6])
max_iterations = 100
batch_size = 1
learning_rate = 0.01
testNN.batchStochasticGradientDescent(training_data, max_iterations, batch_size,
                                     learning_rate)

print testNN.weights
print testNN.biases

# ### Save the weights and biases in JSON
weightsList = testNN.weights
biasesList = testNN.biases
weightsFileName = "../lettersketch/assets/json/UpperCase_StraightLines_weights.json"
biasesFileName = "../lettersketch/assets/json/UpperCase_StraightLines_biases.json"
for weights in weightsList:
  json.dump(weights.tolist(), codecs.open(weightsFileName, 'a', encoding='utf-8'),
    separators=(',', ':'), sort_keys=True, indent=2)

for biases in biasesList:
  json.dump(biases.tolist(), codecs.open(biasesFileName, 'a', encoding='utf-8'),
    separators=(',', ':'), sort_keys=True, indent=2)

# # <font color='green'>Test how well the model is performing</font>
fig = plt.figure()
num_image_rows = np.ceil(np.sqrt(testData.shape[0]))
for i in range(0, testData.shape[0]):
  a = fig.add_subplot(num_image_rows/2, num_image_rows*2, i+1)
  a.set_title(i)
  testplot = plt.imshow(testData[i].reshape(28,28), cmap ="gray")
  plt.axis('off')

result = testNN.forwardCompute(np.reshape(testData[60], (28*28,1)))
letterIndex = np.argmax(result)
print letterIndex
print labelDict.keys()[labelDict.values().index(letterIndex)]


