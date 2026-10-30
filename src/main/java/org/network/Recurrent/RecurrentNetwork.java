package org.network.Recurrent;

import org.network.FeedForward.Activation;

import java.util.Arrays;
import java.util.Random;

import static org.network.util.util.sigmoid;

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

    public float[][] recurrentWeights;

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

        recurrentWeights = new float[hiddenLayerDepth][];
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

            recurrentWeights[i] = new float[nextLayerSize * nextLayerSize];
            recurrentState[i] = new float[nextLayerSize];


        }

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

            float rLimit = (float) Math.sqrt(1.0 / numOutputs);
            for (int j = 0; j < recurrentWeights[layer].length; j++)
                recurrentWeights[layer][j] = random.nextFloat(-rLimit, rLimit);
        }
    }

    public float[] predict(float[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("FATAL: Network expected input of size " + inputSize + " but got " + input.length);
        }

        float[] currentInput = input;
        for (int layer = 0; layer < hiddenLayerDepth; layer++) {
            int inS = layers[layer];
            int outS = layers[layer + 1];
            float[] weights = hiddenLayerWeights[layer];
            float[] recurrentWeight = recurrentWeights[layer];
            float[] bias = hiddenLayerBias[layer];
            float[] outputs = hiddenLayerOutputs[layer];
            float[] prev = recurrentState[layer];
            boolean isOutput = layer == hiddenLayerDepth - 1;

            for (int o = 0; o < outS; o++) {
                float sum = bias[o];
                int wBase = o * inS;
                for (int i = 0; i < inS; i++) sum += currentInput[i] * weights[wBase + i];

                if (!isOutput) {
                    int rBase = o * outS;
                    for (int k = 0; k < outS; k++) sum += prev[k] * recurrentWeight[rBase + k];
                }
                outputs[o] = isOutput ? sigmoid(sum) : (float) Math.tanh(sum);
            }

            if (!isOutput) System.arraycopy(outputs, 0, prev, 0, outS); // update state after the layer
            currentInput = outputs;
        }

        previousOutput = hiddenLayerOutputs[hiddenLayerDepth - 1].clone();
        return previousOutput;
    }

    public void resetState() {
        for (float[] s : recurrentState) Arrays.fill(s, 0f);
        Arrays.fill(previousOutput, 0f);
    }
}
