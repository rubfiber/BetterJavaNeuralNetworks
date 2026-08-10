package org.network.FeedForward;

import org.network.Dataset;
import org.network.neuron.Neuron;
import org.network.util;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

import static org.network.util.sigmoid;
import static org.network.util.softmax;

public class FeedForwardNetwork implements Network, Neuron {
    int inputSize;
    int hiddenLayerDepth;
    int outputSize;
    public float[][] hiddenLayerWeights;
    public float[][] hiddenLayerBias;
    public float[] outputLayerWeights;
    public float[] outputLayerBias;

    float[][] hiddenLayerOutputs;

    float[][] hiddenLayerDeltas;

    public float[][] weightGradients;
    public float[][] biasGradients;

    public FloatArray currentInput;
    public FloatArray currentDesiredOutputs;

    public int totalNetworkLength;
    int[] layers;

    Activation outputActivation =  Activation.SIGMOID;
    Activation hiddenActivation = Activation.TANH;


    /**
     *
     * @param layers This is how the layers will be formatted for your net - you can adjust the number of hidden layers by adding more elements between the first and the last and the neurons per layer by increasing the number of the layer you want to change
     */
    public FeedForwardNetwork(int[] layers) {
        this.layers = layers;

        this.inputSize = layers[0];
        totalNetworkLength = layers.length;
        this.hiddenLayerDepth = totalNetworkLength - 1;
        this.outputSize = layers[totalNetworkLength - 1];

        this.currentInput = new FloatArray(inputSize);
        this.currentDesiredOutputs = new FloatArray(outputSize);

        hiddenLayerWeights = new float[hiddenLayerDepth][];
        hiddenLayerBias = new float[hiddenLayerDepth][];
        hiddenLayerOutputs = new float[hiddenLayerDepth][];
        hiddenLayerDeltas = new float[hiddenLayerDepth][];
        weightGradients = new float[hiddenLayerDepth][];
        biasGradients = new float[hiddenLayerDepth][];

        for (int i = 0; i < hiddenLayerDepth; i++) {
            int currentLayerSize = layers[i];
            int nextLayerSize = layers[i + 1];

            hiddenLayerWeights[i] = new float[currentLayerSize * nextLayerSize];
            hiddenLayerBias[i] = new float[nextLayerSize];
            hiddenLayerOutputs[i] = new float[nextLayerSize];
            hiddenLayerDeltas[i] = new float[nextLayerSize];
            weightGradients[i] = new float[currentLayerSize * nextLayerSize];
            biasGradients[i] = new float[nextLayerSize];
        }

        initializeHiddenLayers();
    }

    /**
     *
     * @param desiredOutputs The target outputs, which is what you expect the network to return
     */
    public void accumulateGradients(float[] input, float[] desiredOutputs) {
        predict(input);

        int lastLayerIdx = hiddenLayerDepth - 1;
        float[] finalOutputs = hiddenLayerOutputs[lastLayerIdx];
        float[] finalDeltas = hiddenLayerDeltas[lastLayerIdx];

        for (int i = 0; i < outputSize; i++) {
            float out = finalOutputs[i];
            float error = out - desiredOutputs[i];
            float derivative = 1.0f;

            if (outputActivation == Activation.SIGMOID) {
                derivative = out * (1.0f - out);
            } else if (outputActivation == Activation.TANH) {
                derivative = 1.0f - (out * out);
            }

            finalDeltas[i] = error * derivative;
        }


        for (int layer = hiddenLayerDepth - 2; layer >= 0; layer--) {
            int currentWidth = layers[layer + 1];
            int nextWidth = layers[layer + 2];

            float[] currentDeltas = hiddenLayerDeltas[layer];
            float[] nextDeltas = hiddenLayerDeltas[layer + 1];
            float[] connectingWeights = hiddenLayerWeights[layer + 1];

            for (int neuron = 0; neuron < currentWidth; neuron++) {
                float blameSum = 0.0f;
                for (int nextNeuron = 0; nextNeuron < nextWidth; nextNeuron++) {
                    int weightIdx = (nextNeuron * currentWidth) + neuron;
                    blameSum += nextDeltas[nextNeuron] * connectingWeights[weightIdx];
                }
                float derivative = findActivationFunction(layer, neuron);

                currentDeltas[neuron] = blameSum * derivative;
            }
        }
        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            float[] targetDeltas = hiddenLayerDeltas[layer];

            int targetWidth = layers[layer + 1];
            int sourceWidth = layers[layer];

            float[] sourceOutputs = (layer == 0) ? input : hiddenLayerOutputs[layer - 1];

            float[] currentWeightGradients = weightGradients[layer];
            float[] currentBiasGradients = biasGradients[layer];
            int weightIndex = 0;

            //bias
            for (int targetNeuron = 0; targetNeuron < targetWidth; targetNeuron++) {
                float targetDelta = targetDeltas[targetNeuron];
                for (int sourceNeuron = 0; sourceNeuron < sourceWidth; sourceNeuron++) {
                    currentWeightGradients[weightIndex++] += targetDelta * sourceOutputs[sourceNeuron];
                }
                currentBiasGradients[targetNeuron] += targetDelta;
            }
        }
    }
    public float[] accumulateGradientsForCnn(float[] input, float[] desiredOutputs) {
        predict(input);

        int lastLayerIdx = hiddenLayerDepth - 1;
        float[] finalOutputs = hiddenLayerOutputs[lastLayerIdx];
        float[] finalDeltas = hiddenLayerDeltas[lastLayerIdx];

        for (int i = 0; i < outputSize; i++) {
            float out = finalOutputs[i];
            float error = out - desiredOutputs[i];
            float derivative = 1.0f;

            if (outputActivation == Activation.SIGMOID) {
                derivative = out * (1.0f - out);
            } else if (outputActivation == Activation.TANH) {
                derivative = 1.0f - (out * out);
            }

            finalDeltas[i] = error * derivative;
        }


        for (int layer = hiddenLayerDepth - 2; layer >= 0; layer--) {
            int currentWidth = layers[layer + 1];
            int nextWidth = layers[layer + 2];

            float[] currentDeltas = hiddenLayerDeltas[layer];
            float[] nextDeltas = hiddenLayerDeltas[layer + 1];
            float[] connectingWeights = hiddenLayerWeights[layer + 1];

            for (int neuron = 0; neuron < currentWidth; neuron++) {
                float blameSum = 0.0f;
                for (int nextNeuron = 0; nextNeuron < nextWidth; nextNeuron++) {
                    int weightIdx = (nextNeuron * currentWidth) + neuron;
                    blameSum += nextDeltas[nextNeuron] * connectingWeights[weightIdx];
                }
                float derivative = findActivationFunction(layer, neuron);

                currentDeltas[neuron] = blameSum * derivative;
            }
        }
        float[] inputErrors = new float[inputSize];
        float[] firstHiddenDeltas = hiddenLayerDeltas[0];
        float[] firstHiddenWeights = hiddenLayerWeights[0];
        int firstHiddenWidth = layers[1];

        // We find out how much blame the flat CNN output deserves
        for (int inNeuron = 0; inNeuron < inputSize; inNeuron++) {
            float inputBlameSum = 0.0f;
            for (int targetNeuron = 0; targetNeuron < firstHiddenWidth; targetNeuron++) {
                int weightIdx = (targetNeuron * inputSize) + inNeuron;
                inputBlameSum += firstHiddenDeltas[targetNeuron] * firstHiddenWeights[weightIdx];
            }
            // Notice there is no derivative applied here!
            // The CNN will apply its own derivative during its backward pass.
            inputErrors[inNeuron] = inputBlameSum;
        }

        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            float[] targetDeltas = hiddenLayerDeltas[layer];

            int targetWidth = layers[layer + 1];
            int sourceWidth = layers[layer];

            float[] sourceOutputs = (layer == 0) ? input : hiddenLayerOutputs[layer - 1];

            float[] currentWeightGradients = weightGradients[layer];
            float[] currentBiasGradients = biasGradients[layer];
            int weightIndex = 0;

            //bias
            for (int targetNeuron = 0; targetNeuron < targetWidth; targetNeuron++) {
                float targetDelta = targetDeltas[targetNeuron];
                for (int sourceNeuron = 0; sourceNeuron < sourceWidth; sourceNeuron++) {
                    currentWeightGradients[weightIndex++] += targetDelta * sourceOutputs[sourceNeuron];
                }
                currentBiasGradients[targetNeuron] += targetDelta;
            }
        }
        return inputErrors;
    }

    private float findActivationFunction(int layer, int neuron) {
        float out = hiddenLayerOutputs[layer][neuron];
        float derivative = 1.0f;

        if (hiddenActivation == Activation.SIGMOID) {
            derivative = out * (1.0f - out);
        } else if (hiddenActivation == Activation.TANH) {
            derivative = 1.0f - (out * out);
        } else if (hiddenActivation == Activation.RELU) {
            derivative = out > 0 ? 1.0f : 0.0f;
        } else if (hiddenActivation == Activation.LEAKY_RELU) {
            derivative = out > 0 ? 1.0f : 0.01f;
        }
        return derivative;
    }


    /**
     *
     * @param learningRate How quickly the network should train - too high, and it might overshoot but too low, and it might not adjust as much - you can tune this to whatever you like
     * @param batchSize How big the batches are - higher time speeds up the time it takes to process during each epoch, but results in a loss in how well the network can generalize new data, outside the training datasets - however, shrinking it uses less memory and adds noise that the network uses to be able to generalize data better on data it hasn't seen, but also has a higher time per epoch
     */
    public void applyBatch(float learningRate, int batchSize) {
        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            float[] weights = hiddenLayerWeights[layer];
            float[] weightGrads = weightGradients[layer];
            for (int weight = 0; weight < weights.length; weight++) {
                weights[weight] -= learningRate * (weightGrads[weight] / batchSize);
                weightGrads[weight] = 0.0f;
            }
            float[] biases = hiddenLayerBias[layer];
            float[] biasGrads = biasGradients[layer];
            for (int bias = 0; bias < biases.length; bias++) {
                biases[bias] -= learningRate * (biasGrads[bias] / batchSize); //same thing but for bias
                biasGrads[bias] = 0.0f;
            }
        }
    }

    /**
     *
    * @param dataset The dataset that the network is trained on. Includes the input data and output data, both in the format float[][], where the first value is the train index and the second array is for what should be inputted
     * @param epochs The number of epochs, or training loops, the network goes through - lower number reduces the time it takes to train the network, but causes underfitting. This is where the network fails to learn the rules of the data and causes low accuracy. However, more epochs has a higher training accuracy, but causes overfitting, where the model memorizes the exact details of the training data and becomes too used to it
     * @param batchSize How big the batches are - higher time speeds up the time it takes to process during each epoch, but results in a loss in how well the network can generalize new data, outside the training datasets - however, shrinking it uses less memory and adds noise that the network uses to be able to generalize data better on data it hasn't seen, but also has a higher time per epoch
     * @param learningRate How quickly the network should train - too high, and it might overshoot but too low, and it might not adjust as much - you can tune this to whatever you like
     */
    @Override
    public void trainDataset(Dataset dataset, int epochs, int batchSize, float learningRate) {
        float[][] input = dataset.getInputData();
        float[][] desiredOutputs = dataset.getOutputData();
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
            for (int i = 0; i < numSamples; i += batchSize) {
                int trueBatchSize = Math.min(batchSize, numSamples - i);
                for (int j = 0; j < trueBatchSize; j++) {
                    int sampleIndex = indices[i + j]; //train with these
                    accumulateGradients(input[sampleIndex], desiredOutputs[sampleIndex]);
                }
                applyBatch(learningRate, trueBatchSize);
            }

        }
    }
    public void setOutputActivation(Activation activation) {
        outputActivation = activation;
    }
    public void setHiddenActivations(Activation activation) {
        hiddenActivation = activation;
    }
    /**
     * The model makes a prediction based on an input, using math to calculate its output from the weights and biases
     * @param input The data you want the model to make a prediction from
     * @return The prediction based on the weights and biases provided
     */
    @Override
    public float[] predict(float[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("FATAL: Network expected input of size " + inputSize + " but got " + input.length);
        }

        float[] currentInput = input;
        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            int inputS = layers[layer];
            int outputS = layers[layer + 1];

            float[] weights = hiddenLayerWeights[layer];
            float[] bias = hiddenLayerBias[layer];
            float[] outputs = hiddenLayerOutputs[layer];

            int wIndex = 0;

            for (int outNeuron = 0; outNeuron < outputS; outNeuron++) {
                float sum = 0.0f;

                for (int inNeuron = 0; inNeuron < inputS; inNeuron++) {
                    sum += currentInput[inNeuron] * weights[wIndex++];
                }
                sum += bias[outNeuron];
                if (layer == hiddenLayerDepth - 1) {

                    if (outputActivation == Activation.SOFTMAX || outputActivation == Activation.LINEAR) {
                        outputs[outNeuron] = sum; // Softmax and Linear need the raw sum!
                    } else if (outputActivation == Activation.SIGMOID) {
                        outputs[outNeuron] = util.sigmoid(sum);
                    } else if (outputActivation == Activation.TANH) {
                        outputs[outNeuron] = (float) Math.tanh(sum); // Fixed Math.tanh!
                    } else {
                        outputs[outNeuron] = sum; // Safe fallback
                    }
                } else {

                    if (hiddenActivation == Activation.RELU) {
                        outputs[outNeuron] = util.relu(sum);
                    } else if (hiddenActivation == Activation.LEAKY_RELU) {
                        outputs[outNeuron] = util.leakyRelu(sum);
                    } else if (hiddenActivation == Activation.TANH) {
                        outputs[outNeuron] = (float) Math.tanh(sum);
                    } else {
                        outputs[outNeuron] = util.sigmoid(sum);
                    }
                }
            }

            currentInput = outputs;
        }
        return (outputActivation == Activation.SOFTMAX) ? softmax(hiddenLayerOutputs[hiddenLayerDepth - 1]) : hiddenLayerOutputs[hiddenLayerDepth - 1];
    }

    /**
     * Loads the weights from a .bin file
     * @param weightFilePath - The file path leading to the weights you want to store - you can also use relative paths ( /src/...)
     * @throws IOException If there was an error loading the weights
     */
    public void loadWeights(String weightFilePath) throws IOException {
        Path weightsPath = Paths.get(weightFilePath);

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(weightsPath.toFile())))) {
            int length = dis.readInt();

            float[] weightData = new float[length];
            for (int i = 0; i < length; i++) {
                weightData[i] = dis.readFloat();
            }

            this.hiddenLayerWeights = util.deflattenWeights(weightData, this.layers);
        }
    }

    /**
     * Loads the biases from a .bin file
     * @param biasFilePath - The file path leading to the biases you want to store - you can also use relative paths ( /src/...)
     * @throws IOException If there was an error loading the biases
     */

    public void loadBias(String biasFilePath) throws IOException {
        Path biasPath = Paths.get(biasFilePath);

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(biasPath.toFile())))) {
            int length = dis.readInt();

            float[] biasData = new float[length];
            for (int i = 0; i < length; i++) {
                biasData[i] = dis.readFloat();
            }

            this.hiddenLayerBias = util.deflattenBias(biasData, this.layers);
        }
    }

    /**
     * Saves the state of the network to a .bin file
     * @param weightFilePath The file path leading to the weights you want to store - you can also use relative paths ( /src/...)
     * @param biasFilePath The file path leading to the biases you want to store - you can also use relative paths ( /src/...)
     */
    public void saveState(String weightFilePath, String biasFilePath) {
        float[] flattenedWeights = util.flatten(this.hiddenLayerWeights);
        float[] flattenedBias = util.flatten(this.hiddenLayerBias);

        writeWeightsAndBias(weightFilePath, biasFilePath, flattenedWeights, flattenedBias);
    }

    /**
     * Write weights and biases to a .bin file
     * @param weightFilePath The file path leading to the weights you want to store - you can also use relative paths ( /src/...)
     * @param biasFilePath The file path leading to the biases you want to store - you can also use relative paths ( /src/...)
     * @param flattenedWeights The weights tou want to store (flattened)
     * @param flattenedBias The weights you want to store (flattened)
     */
    private static void writeWeightsAndBias(String weightFilePath, String biasFilePath, float[] flattenedWeights, float[] flattenedBias) {
        Path weightsPath = Paths.get(weightFilePath);
        Path biasPath = Paths.get(biasFilePath);
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(weightsPath.toFile())))) {

            dos.writeInt(flattenedWeights.length);
            for (float value : flattenedWeights) {
                dos.writeFloat(value);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(biasPath.toFile())))) {

            dos.writeInt(flattenedBias.length);
            for (float value : flattenedBias) {
                dos.writeFloat(value);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Initializes the hidden layers using Glorot initialization
     */
    @Override
    public void initializeHiddenLayers() {
        final Random random = new Random();

        for (int layer = 0; layer < hiddenLayerDepth; layer++) {

            Arrays.fill(hiddenLayerBias[layer], 0.0f);

            int numInputs = layers[layer];
            int numOutputs = layers[layer + 1];

            float limit = (float) Math.sqrt(6.0 / (numInputs + numOutputs));

            float[] weightsRow = hiddenLayerWeights[layer];
            for (int j = 0; j < weightsRow.length; j++) {
                weightsRow[j] = random.nextFloat(-limit, limit);
            }
        }
    }

    /**
     * Initializes the output layers using Glorot initialization
     */
    @Override
    public void initializeOutputLayer() {
        final Random random = new Random();
        Arrays.fill(outputLayerBias, 0);

        float outputLayerLimit = (float) Math.sqrt(6.0 / (layers[totalNetworkLength - 2] + outputSize));
        for (int i = 1; i < outputSize; i++) {
            outputLayerWeights[i] =  random.nextFloat(-outputLayerLimit, outputLayerLimit);
        }
    }
}