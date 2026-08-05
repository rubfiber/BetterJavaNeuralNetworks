# Java Neural nets

These are neural networks written in Java. Currently, there is only 1 architecture (Feedforward) but I am working on more types.

## Installation

You can clone this repository with this command:\
```git clone https://github.com/rubfiber/BetterJavaNeuralNetworks/```

## Usage
You can use the ```org.network.FeedForward.feedForwardNetwork``` class to create a network. \
Format the constructor like this: ```FeedForwardNetwork network = new FeedForwardNetwork(inputSize, hiddenLayerWidth, hiddenLayerDepth, outputSize), where:
- inputSize is the size of the inputs, which also determines the number of units/neurons in the input layer
- hiddenLayerWidth is the width of the hidden layers, determining the number of neurons per layer (will be replaced with an array later)
- hiddenLayerDepth is the number of hidden layers
- outputSize is the size of the output

### Predictions
Use the ```network.predict(float[] input) ```function, where the input is what is being inputted into the network.\
### Training
Using the ```network.trainDataset(trainDataset(float[][] input, float[][] desiredOutputs, int epochs, int batchSize, float learningRate)```, you can train the network, where:
- input is the list of all inputs from the dataset (each element is its own array of floats)
- desiredOutputs is the outputs that you want to see come out of the network, where each corresponds to one input.
- epochs is the number of training loops that the network goes through. Too little results in underfitting, where the model has not learned enough, but too much causes overfitting, causing the network to become too used to the training data
- batchSize is the size of the batches, since we are using mini-batch gradient descent. You can always fine-tune it
- learningRate is how much the neurons respond to changes. You can also fine tune this value.
### Saving to a file
The network allows you to save its state to a file, holding the weights and biases. \
First, you need to create a file called ```hiddenWeights.bin```, ```hiddenBias.bin```, ```outputWeights.bin```, and ```outputBias.bin```.\
Use the ```saveState(float[][] hiddenWeights, float[][] hiddenBias, float[] outputWeights, float[] outputBias, String weightFilePath, String biasFilePath, String outputWeightFilePath, String outputBiasFilePath)```, where:
- hiddenWeights is the hidden layer weight array you want to save, using network.hiddenLayerWeights.
- hiddenBias is the hidden layer bias array you want to save, using network.hiddenLayerBias.
- outputWeights is the output weight array you want to save, using network.outputLayerWeights.
- outputBias is the output layer bias array you want to save, using network.outputLayerBias.
- The Strings are their respective paths, e.g. ```/home/user/network/hiddenWeights.bin```.
### Loading a network state
There are 2 methods to load the network state, one for loading weights and the other for loading bias. \
The one for loading weights is ```network.loadWeights(String hiddenWeightFilePath, String outputWeightFilePath)```, where:
- hiddenWeightFilePath is the path of the hidden weights file, which is saved as a ```.bin``` file.
- outputWeightFilePath is the path of the output layer weights file, which is saved  as a ```.bin``` file. 



The one for loading biases is ```network.loadBias(String hiddenBiasFilePath, String outputBiasFilePath)```, where:
- hiddenBiasFilePath is the path of the hidden bias file, which is saved as a ```.bin``` file.
- outputBiasFilePath is the path of the output layer bias file, which is saved  as a ```.bin``` file.