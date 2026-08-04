# Java Neural nets

These are neural networks written in Java. Currently there is only 1 architecture (Feedforward) but I am working on more types.

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
