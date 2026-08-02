package org.network.FeedForward;

import org.network.neuron.Neuron;

import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

public class FeedForwardNetwork implements Network, Neuron{
    int inputSize;
    int hiddenLayerWidth;
    int hiddenLayerDepth;
    int outputSize;
    public float[][] hiddenLayerWeights;
    public float[][] hiddenLayerBias;
    public float[] outputLayerWeights;
    public float[] outputLayerBias;

    float[][] hiddenLayerOutputs;

    float[][] hiddenLayerDeltas;
    float[] outputLayerDeltas;

    public FeedForwardNetwork(int inputSize, int hiddenLayerWidth, int hiddenLayerDepth, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayerWidth = hiddenLayerWidth;
        this.hiddenLayerDepth = hiddenLayerDepth;
        this.outputSize = outputSize;

        hiddenLayerWeights = new float[hiddenLayerDepth][];

        hiddenLayerWeights[0] = new float[inputSize * hiddenLayerWidth]; //since first layer inputs from input layer
        for (int i = 1; i < hiddenLayerDepth; i++) {
            hiddenLayerWeights[i] = new float[hiddenLayerWidth * hiddenLayerWidth];
        }

        hiddenLayerWeights[hiddenLayerDepth - 1] = new float[hiddenLayerWidth * outputSize];

        hiddenLayerBias = new float[hiddenLayerDepth][hiddenLayerWidth];

        outputLayerWeights = new float[outputSize];
        outputLayerBias = new float[outputSize];

        initializeOutputLayer();

        initializeHiddenLayers();

        hiddenLayerOutputs = new float[hiddenLayerDepth][];
        for (int i = 0; i < hiddenLayerDepth - 1; i++) {
            hiddenLayerOutputs[i] = new float[hiddenLayerWidth];
        }
        hiddenLayerOutputs[hiddenLayerDepth - 1] = new float[outputSize];

        hiddenLayerDeltas = new float[hiddenLayerDepth][];
        for  (int i = 0; i < hiddenLayerDepth-1; i++) {
            hiddenLayerDeltas[i] = new float[hiddenLayerWidth];
        }
        hiddenLayerDeltas[hiddenLayerDepth - 1] = new float[outputSize];
        outputLayerDeltas = new float[outputSize];

    }

    public void trainLoop(float[] test, float[] desiredOutputs, float learningRate) {
        //forward pass
        float[] output = predict(test); //fill up hiddenLayerOutputs and output
        for (int i =  0; i < outputSize; i++) {
            float neuronOutput = output[i];
            outputLayerDeltas[i] = (neuronOutput - desiredOutputs[i]) * (1.0f - neuronOutput * neuronOutput);
        }

        //propagate error backwards
        for (int layer = hiddenLayerDepth - 2; layer >= 0; layer--) {
            float[] layerDeltas = hiddenLayerDeltas[layer]; //the size can change if the size of the layer changes
            float[] nextLayerDeltas;
            float[] connectingWeights = hiddenLayerWeights[layer + 1];

            int currentWidth = layerDeltas.length;
            int nextWidth;
            if (layer == hiddenLayerDepth - 2) {
                nextLayerDeltas = outputLayerDeltas;
                nextWidth = outputSize;
            } else {
                nextLayerDeltas = hiddenLayerDeltas[layer + 1];
                nextWidth = hiddenLayerWidth;
            }
            //get deltas
            for (int neuron = 0; neuron < currentWidth; neuron++) {
                float blameSum = 0;
                for (int nextNeuron = 0; nextNeuron < nextWidth; nextNeuron++) {
                    int weight = (nextNeuron * currentWidth) + neuron;
                    blameSum += nextLayerDeltas[nextNeuron] * connectingWeights[weight];
                }
                float layerOutput = hiddenLayerOutputs[layer][neuron];
                layerDeltas[neuron] = blameSum * (layerOutput * (1.0f - layerOutput));
            }
        }
        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            float[] targetDeltas;
            float[] sourceOutputs;
            float[] currentWeights = hiddenLayerWeights[layer];
            float[] currentBiases = hiddenLayerBias[layer];

            int targetWidth;
            int sourceWidth;

            // check location (start, end, or middle)
            if (layer == 0) {
                targetDeltas = hiddenLayerDeltas[0];
                sourceOutputs = test;
                targetWidth = hiddenLayerWidth;
                sourceWidth = inputSize;
            } else if (layer == hiddenLayerDepth - 1) {
                targetDeltas = outputLayerDeltas;
                sourceOutputs = hiddenLayerOutputs[layer - 1];
                targetWidth = outputSize;
                sourceWidth = hiddenLayerWidth;
            } else {
                targetDeltas = hiddenLayerDeltas[layer];
                sourceOutputs = hiddenLayerOutputs[layer - 1];
                targetWidth = hiddenLayerWidth;
                sourceWidth = hiddenLayerWidth;
            }

            int weightIndex = 0;

            // LINEAR MEMORY ACCESS: Target is outer loop, Source is inner loop
            for (int targetNeuron = 0; targetNeuron < targetWidth; targetNeuron++) {
                float targetDelta = targetDeltas[targetNeuron];

                for (int sourceNeuron = 0; sourceNeuron < sourceWidth; sourceNeuron++) {
                    float adjust = learningRate * targetDelta * sourceOutputs[sourceNeuron];
                    currentWeights[weightIndex] -= adjust;
                    weightIndex++;
                }

                // Update the bias for the target neuron
                currentBiases[targetNeuron] -= learningRate * targetDelta;
            }
        }
    }

    @Override
    public void train(float[][] input, float[][] desiredOutputs, int epochs, float learningRate) {
        int numSamples = input.length;

        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }

        Random random = new Random();
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {
                int swap = random.nextInt(numSamples);
                int temp = indices[i];
                indices[i] = indices[swap];
                indices[swap] = temp; //swap temp and swap (data is shuffled now)
            }
            for (int i = 0; i < numSamples; i++) {
                float[] currentInput = input[i];
                float[] currentExpectedOutput = desiredOutputs[i];
                trainLoop(currentInput, currentExpectedOutput, learningRate);
            }

        }
    }

    @Override
    public float[] predict(float[] input) {
        assert input.length == inputSize : "Input length has to be the same as input layer size";
        //input layer has no math, it just passes values to hidden layers
        //first layer
        float[] firstLayerWeights = hiddenLayerWeights[0];
        float[] firstLayerBias = hiddenLayerBias[0];
        float[] firstLayerOutputs =  hiddenLayerOutputs[0];
        int weightIndex = 0;
        for (int neuron = 0; neuron < hiddenLayerWidth; neuron++) {
            float sum = 0.0f;

            for (int in = 0; in < inputSize; in++) {
                sum += input[in] * firstLayerWeights[weightIndex++];
            }

            sum += firstLayerBias[neuron];

            firstLayerOutputs[neuron] = sigmoid(sum);

            hiddenLayerOutputs[0] = firstLayerOutputs;
            if (Float.isNaN(firstLayerOutputs[neuron]) || Float.isInfinite(firstLayerOutputs[neuron])) {
                throw new IllegalStateException("NaN or Infinity in compute(). Sum=" + sum);
            }
        }
        // hidden layers
        for (int layer = 1; layer < hiddenLayerDepth-1; layer++) {
            float[] layerWeights = hiddenLayerWeights[layer];
            float[]  layerBias = hiddenLayerBias[layer];
            float[] layerOutputs =  hiddenLayerOutputs[layer];
            float[] layerInput = hiddenLayerOutputs[layer-1];
            int middleWeightIndex = 0;
            for (int neuron = 0; neuron < hiddenLayerWidth; neuron++) {
                float sum = 0.0f;
                for (float v : layerInput) {
                    sum += v * layerWeights[middleWeightIndex++];
                }
                sum += layerBias[neuron];

                layerOutputs[neuron] = sigmoid(sum);
            }

        }
        int lastLayerIdx = hiddenLayerDepth - 1;
        float[] outLayerWeights = hiddenLayerWeights[lastLayerIdx];
        float[] outLayerBias = hiddenLayerBias[lastLayerIdx];
        float[] outLayerOutputs = hiddenLayerOutputs[lastLayerIdx];
        float[] lastHiddenOutputs = hiddenLayerOutputs[lastLayerIdx - 1];
        int outWeightIndex = 0;

        for (int neuron = 0; neuron < outputSize; neuron++) {
            float sum = 0.0f;
            for (int in = 0; in < hiddenLayerWidth; in++) {
                sum += lastHiddenOutputs[in] * outLayerWeights[outWeightIndex++];
            }
            sum += outLayerBias[neuron];

            //math.tanh since output layer
            outLayerOutputs[neuron] = (float) Math.tanh(sum);
        }

        return outLayerOutputs;


    }

    @Override
    public void loadState(File file) {

    }

    @Override
    public void saveState(Path path) {

    }
//Neuron functions
    @Override

    public void initializeHiddenLayers() {
        final Random random = new Random();
        Arrays.fill(hiddenLayerBias[0], 0);

        float firstLayerLimit = (float) Math.sqrt(6.0 / (inputSize + hiddenLayerWidth));
        float[] firstRow = hiddenLayerWeights[0];
        for (int i = 0; i < firstRow.length; i++) { //weight size is input size
            hiddenLayerWeights[0][i] = random.nextFloat(-firstLayerLimit, firstLayerLimit);
        }


        for (int i = 1; i < hiddenLayerDepth - 1; i++) {
            Arrays.fill(hiddenLayerBias[i], 0);

            int numInputs = hiddenLayerWeights[i - 1].length;
            int numOutputs = hiddenLayerWeights[i + 1].length;
            float middleLimit = (float) Math.sqrt(6.0 / (numInputs + numOutputs));

            float[] weightsRow = hiddenLayerWeights[i];
            int length = weightsRow.length;

            for (int j = 0; j < length; j++) {
                weightsRow[j] =  random.nextFloat(-middleLimit, middleLimit);
            }
        }
        int lastIndex = hiddenLayerDepth-1;
        float lastLayerLimit = (float) Math.sqrt(6.0 / (hiddenLayerDepth + outputSize));
        float[] lastRow = hiddenLayerWeights[lastIndex];
        Random random1 = new Random();
        for (int i = 0; i < lastRow.length; i++) {
            hiddenLayerWeights[hiddenLayerDepth - 1][i] = random1.nextFloat(-lastLayerLimit, lastLayerLimit);
        }

    }
    @Override
    public void initializeOutputLayer() {
        final Random random = new Random();
        Arrays.fill(outputLayerBias, 0);

        float outputLayerLimit = (float) Math.sqrt(6.0 / (hiddenLayerWidth + outputSize));
        for (int i = 1; i < outputSize; i++) {
            outputLayerWeights[i] =  random.nextFloat(-outputLayerLimit, outputLayerLimit);
        }
    }

    @Override
    public float neuronCompute(float[] weights, float bias, float[] input, boolean isOutLayer) {
        assert input != null : "Input is null";
        assert weights != null : "Weights is null";
        assert input.length == weights.length;

        float sum = 0.0f;
        for (int i = 0; i < input.length; i++) {
            sum += input[i] * weights[i];
        }
        sum += bias;
        float activated;
        if (isOutLayer) {
            activated = (float) Math.tanh(sum);
        } else {
            activated = sigmoid(sum);
        }

        if (Float.isNaN(activated) || Float.isInfinite(activated))
            throw new IllegalStateException("NaN or Infinity in compute(). Sum=" + sum);

        return activated;
    }

    @Override
    public float getWeights(int neuronIdx) {
        return 0;
    }

    @Override
    public float getBias(int neuronIdx) {
        return 0;
    }

    @Override
    public void setWeights(int neuronIdx, float value) {

    }

    @Override
    public void setBias(int neuronIdx, float value) {

    }

    private static float sigmoid(float x) {
        if (x >= 0) {
            float z = (float) Math.exp(-x);
            return 1 / (1 + z);
        } else {
            float z = (float) Math.exp(x);
            return z / (1 + z);
        }
    }
}
