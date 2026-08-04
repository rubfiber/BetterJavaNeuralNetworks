package org.network.FeedForward;

import org.network.neuron.Neuron;

import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

//import uk.ac.manchester.tornado.api.TaskGraph;
//import uk.ac.manchester.tornado.api.annotations.Parallel;
//import uk.ac.manchester.tornado.api.enums.DataTransferMode;
//import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import static org.network.util.sigmoid;

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

    public float[][] weightGradients;
    public float[][] biasGradients;

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

        weightGradients = new float[hiddenLayerDepth][];
        weightGradients[0] = new float[inputSize * hiddenLayerWidth];
        for (int i = 1; i < hiddenLayerDepth - 1; i++) {
            weightGradients[i] = new float[hiddenLayerWidth * hiddenLayerWidth];
        }
        weightGradients[hiddenLayerDepth - 1] = new float[hiddenLayerWidth * outputSize];

        biasGradients = new float[hiddenLayerDepth][hiddenLayerWidth];
    }

    public void accumulateGradients(float[] test) {
        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            float[] targetDeltas;
            float[] sourceOutputs;

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


            float[] currentWeightGradients = weightGradients[layer];
            float[] currentBiasGradients = biasGradients[layer];
            int weightIndex = 0;
            //bias
            for (int targetNeuron = 0; targetNeuron < targetWidth; targetNeuron++) {
                float targetDelta = targetDeltas[targetNeuron];
                //weights
                for (int sourceNeuron = 0; sourceNeuron < sourceWidth; sourceNeuron++) {
                    float gradient = targetDelta * sourceOutputs[sourceNeuron];
                    //accumulation
                    currentWeightGradients[weightIndex] += gradient;
                    weightIndex++;
                }
                //accumulate bias
                currentBiasGradients[targetNeuron] += targetDelta;
            }
        }
    }
    public void applyBatch(float learningRate, int batchSize) {
        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            float[] weights = hiddenLayerWeights[layer];
            float[] weightGrads = weightGradients[layer];
            for (int weight = 0; weight < weights.length; weight++) {
                weights[weight] -= learningRate * (weightGrads[weight] / batchSize);
                weightGrads[weight] = 0.0f;
            }
            float [] biases = hiddenLayerBias[layer];
            float[] biasGrads = biasGradients[layer];
            for (int bias = 0; bias < biases.length; bias++) {
                biases[bias] -= learningRate * (biasGrads[bias] / batchSize); //same thing but for bias
            }
        }
    }

    @Override
    public void trainDataset(float[][] input, float[][] desiredOutputs, int epochs, int batchSize, float learningRate) {
        int numSamples = input.length;

        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }

        Random random = new Random();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {
                int swap = random.nextInt(numSamples); //prevents network from memorizing order
                int temp = indices[i];
                indices[i] = indices[swap];
                indices[swap] = temp; //swap temp and swap (data is shuffled now)
            }
            for (int i = 0; i < numSamples; i+=batchSize) {
                int trueBatchSize = Math.min(batchSize, numSamples - i);

                for (int j = 0; j < trueBatchSize; j++) {
                    int sampleIndex= indices[i+j]; //train with these
                    accumulateGradients(input[sampleIndex]);
                }
                applyBatch(learningRate, trueBatchSize);
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


}
