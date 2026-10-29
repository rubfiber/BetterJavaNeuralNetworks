package org.network.Recurrent;

import org.network.FeedForward.Activation;
import org.network.util.util;

import java.util.Arrays;
import java.util.Random;

public class RecurrentNetwork {
    public final int[] layers;
    public float[][] recurrentState;
    public float[] previousOutput;
    public final int hiddenLayerDepth;
    public final int totalNetworkLength;
    private final int outputSize;
    public final int inputSize;

    public float[][] hiddenLayerWeights;
    public float[][] hiddenLayerBias;
    public float[][] hiddenLayerOutputs;
    public float[][] hiddenLayerDeltas;
    public float[][] weightGradients;
    public float[][] biasGradients;
    public Activation[] outputActivations;
    public Activation[][] hiddenActivations;

    public enum RecurrentTypes {
        MANY_TO_ONE,
        ONE_TO_MANY,
        MANY_TO_MANY
    }

    public RecurrentNetwork(int[] layers) { //input
        this.layers = layers;
        totalNetworkLength = layers.length;
        this.inputSize = layers[0];
        hiddenLayerDepth = totalNetworkLength - 1;
        this.outputSize = layers[totalNetworkLength - 1];


        recurrentState = new float[hiddenLayerDepth][];

        hiddenLayerWeights = new float[hiddenLayerDepth][];
        hiddenLayerBias = new float[hiddenLayerDepth][];
        hiddenLayerOutputs = new float[hiddenLayerDepth][];
        hiddenLayerDeltas = new float[hiddenLayerDepth][];
        weightGradients = new float[hiddenLayerDepth][];
        biasGradients = new float[hiddenLayerDepth][];

        outputActivations = new Activation[outputSize];
        hiddenActivations = new Activation[hiddenLayerDepth][];

        Arrays.fill(outputActivations, Activation.SIGMOID);

        for (int i = 0; i < hiddenLayerDepth; i++) {
            int currentLayerSize = layers[i];
            int nextLayerSize = layers[i + 1];

            hiddenLayerWeights[i] = new float[currentLayerSize * nextLayerSize];
            hiddenLayerBias[i] = new float[nextLayerSize];
            hiddenLayerOutputs[i] = new float[nextLayerSize];
            hiddenLayerDeltas[i] = new float[nextLayerSize];
            weightGradients[i] = new float[currentLayerSize * nextLayerSize];
            biasGradients[i] = new float[nextLayerSize];
            hiddenActivations[i] = new Activation[nextLayerSize];
            Arrays.fill(hiddenActivations[i],  Activation.LEAKY_RELU);

            recurrentState[i] = new float[currentLayerSize];
            Arrays.fill(recurrentState[i], 0f);

        }
        recurrentState[totalNetworkLength - 1] = new float[outputSize];
        Arrays.fill(recurrentState[totalNetworkLength-1], 0f);

        previousOutput = new  float[outputSize];
        Arrays.fill(previousOutput, 0f); //for one to many

        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            final Random random = new Random();

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

    public float[] predict(float[] input, RecurrentTypes type) {
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

            for (int outNeuron = 0; outNeuron < outputS; outNeuron++) { //next layer size
                float sum = 0.0f;

                for (int inNeuron = 0; inNeuron < inputS; inNeuron++) {
                    sum += currentInput[inNeuron] * weights[wIndex++];
                    if (type == RecurrentTypes.MANY_TO_MANY) sum += recurrentState[layer][inNeuron] * bias[wIndex++];
                }
                sum += bias[outNeuron];
                recurrentState[layer][outNeuron] = sum;

                outputs[outNeuron] = (float) Math.tanh(sum);

                if (layer == hiddenLayerDepth - 1) { // output layer
                    previousOutput = outputs;
                }
            }

            currentInput = outputs;
        }


        return hiddenLayerOutputs[hiddenLayerDepth - 1];
    }
}
