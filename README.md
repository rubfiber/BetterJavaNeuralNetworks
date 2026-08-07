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
Using the ```network.trainDataset(trainDataset(Dataset dataset, int epochs, int batchSize, float learningRate)```, you can train the network, where:
- dataset is the dataset that the network is trained on. Includes the input data and output data, both in the format float[][], where the first value is the train index and the second array is for what should be inputted
- epochs is the number of training loops that the network goes through. Too little results in underfitting, where the model has not learned enough, but too much causes overfitting, causing the network to become too used to the training data
- batchSize is the size of the batches, since we are using mini-batch gradient descent. You can always fine-tune it
- learningRate is how much the neurons respond to changes. You can also fine tune this value.
### Saving to a file
The network allows you to save its state to a file, holding the weights and biases. \
First, you need to create 2 ```.bin``` files, named whatever you want - one for  weights and one for bias, e.g. ```weights.bin``` and ```bias.bin```, \
Use the ```saveState(String weightFilePath, String biasFilePath)```, where:
- weightFilePath is the path to the weights file (as a ```.bin``` file), e.g. ```/home/user/network/weights.bin```.
- biasFilePath is the path to the bias file (as a ```.bin``` file), e.g. ```/home/user/network/bias.bin```.
### Loading a network state
There are 2 methods to load the network state, one for loading weights and the other for loading bias. \
The one for loading weights is ```loadWeights(String weightFilePath)```, where:
- weightFilePath is the path of the weights file, which is saved as a ```.bin``` file.


The one for loading biases is ```network.loadBias(String BiasFilePath)```, where:
- BiasFilePath is the path of the bias file, which is saved as a ```.bin``` file.
