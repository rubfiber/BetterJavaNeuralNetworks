package org.network.Recurrent;

import org.network.FeedForward.Activation;

import java.util.Arrays;
import java.util.Random;

import static org.network.util.util.sigmoid;

public class LSTM {
        private final int[] layers;
        private final int totalNetworkLength;
        private final int hiddenLayerDepth;
        private final int inputSize;
        private final int outputSize;

        private final float[][] hiddenState;
        private final float[][] cellState;

        private final float[][] forgetGateWeights;
        private final float[][] inputGateWeights;
        private final float[][] candidateWeights;
        private final float[][] outputGateWeights;

        private final float[][] forgetGateBias;
        private final float[][] inputGateBias;
        private final float[][] candidateBias;
        private final float[][] outputGateBias;

        private final float[][] forgetWeightGradients;
        private final float[][] inputWeightGradients;
        private final float[][] candidateWeightGradients;
        private final float[][] outputWeightGradients;

        private final int lstmLayerDepth;
        private final float[] denseWeights;
        private final float[] denseBias;
        private final float[] denseWeightGradients;
        private final float[] denseBiasGradients;
        private final Activation[] denseOutputActivations;

        private final float[] previousOutput;

        public LSTM(int[] layers) {
            this.layers = layers;
            this.totalNetworkLength = layers.length;
            this.inputSize = layers[0];
            this.hiddenLayerDepth = totalNetworkLength - 1;
            this.outputSize = layers[totalNetworkLength - 1];
            this.lstmLayerDepth = totalNetworkLength - 2;


            hiddenState = new float[hiddenLayerDepth][];
            cellState = new float[hiddenLayerDepth][];

            // gates
            forgetGateWeights = new float[hiddenLayerDepth][];
            inputGateWeights = new float[hiddenLayerDepth][];
            candidateWeights = new float[hiddenLayerDepth][];
            outputGateWeights = new float[hiddenLayerDepth][];

            forgetGateBias = new float[hiddenLayerDepth][];
            inputGateBias = new float[hiddenLayerDepth][];
            candidateBias = new float[hiddenLayerDepth][];
            outputGateBias = new float[hiddenLayerDepth][];

            forgetWeightGradients = new float[hiddenLayerDepth][];
            inputWeightGradients = new float[hiddenLayerDepth][];
            candidateWeightGradients = new float[hiddenLayerDepth][];
            outputWeightGradients = new float[hiddenLayerDepth][];

            // dense/output layer
            int lastHiddenSize = layers[totalNetworkLength - 2]; // Input to the dense layer

            denseWeights = new float[lastHiddenSize * outputSize];
            denseBias = new float[outputSize]; // Defaults to 0.0f

            denseWeightGradients = new float[lastHiddenSize * outputSize];
            denseBiasGradients = new float[outputSize];

            denseOutputActivations = new Activation[outputSize];
            Arrays.fill(denseOutputActivations, Activation.SIGMOID);

            previousOutput = new float[outputSize];

            final Random random = new Random();

            for (int i = 0; i < hiddenLayerDepth-1; i++) {
                int numInputs = layers[i];
                int numOutputs = layers[i + 1];

                // combined input size when adding/concatanating
                int combinedInputSize = numInputs + numOutputs;
                int totalGateWeights = combinedInputSize * numOutputs;

                hiddenState[i] = new float[numOutputs];
                cellState[i] = new float[numOutputs];

                forgetGateWeights[i] = new float[totalGateWeights];
                inputGateWeights[i] = new float[totalGateWeights];
                candidateWeights[i] = new float[totalGateWeights];
                outputGateWeights[i] = new float[totalGateWeights];

                forgetWeightGradients[i] = new float[totalGateWeights];
                inputWeightGradients[i] = new float[totalGateWeights];
                candidateWeightGradients[i] = new float[totalGateWeights];
                outputWeightGradients[i] = new float[totalGateWeights];

                forgetGateBias[i] = new float[numOutputs];
                inputGateBias[i] = new float[numOutputs];
                candidateBias[i] = new float[numOutputs];
                outputGateBias[i] = new float[numOutputs];

                Arrays.fill(forgetGateBias[i], 1.0f);

                float limit = (float) Math.sqrt(6.0 / (combinedInputSize + numOutputs));

                for (int w = 0; w < totalGateWeights; w++) {
                    forgetGateWeights[i][w] = random.nextFloat(-limit, limit);
                    inputGateWeights[i][w] = random.nextFloat(-limit, limit);
                    candidateWeights[i][w] = random.nextFloat(-limit, limit);
                    outputGateWeights[i][w] = random.nextFloat(-limit, limit);
                }
            }
            float outputLimit = (float) Math.sqrt(6.0 / (lastHiddenSize + outputSize));
            for (int w = 0; w < denseWeights.length; w++) {
                denseWeights[w] = random.nextFloat(-outputLimit, outputLimit);
            }
        }
        public float[] predict(float[] input) {
            if (input.length != inputSize) {
                throw new IllegalArgumentException("FATAL: Network expected input of size " + inputSize + " but got " + input.length);
            }
            float[] currentLayerInput = input;
            for (int layer = 0; layer <= lstmLayerDepth; layer++) {
                int numInputs = layers[layer];
                int numHidden = layers[layer + 1];
                int combinedInputSize = numInputs + numHidden;

                float[] concatenatedInput = new float[combinedInputSize];
                System.arraycopy(hiddenState[layer], 0, concatenatedInput, 0, numHidden);
                System.arraycopy(currentLayerInput, 0, concatenatedInput, numHidden, numInputs);

                for (int j = 0; j < numHidden; j++) {
                    int offset = j * combinedInputSize;
                    float forgetSum = forgetGateBias[layer][offset];
                    float inputSum = inputGateBias[layer][offset];
                    float candidateSum = candidateBias[layer][offset];
                    float outputSum = outputGateBias[layer][offset];

                    for (int k = 0; k < combinedInputSize; k++) {
                        int weightIdx = offset + k;
                        float combinedVal = concatenatedInput[k];
                        forgetSum += forgetGateWeights[layer][weightIdx] * combinedVal;
                        inputSum += inputGateWeights[layer][weightIdx] * combinedVal;
                        candidateSum += candidateWeights[layer][weightIdx] * combinedVal;
                        outputSum += outputGateWeights[layer][weightIdx] * combinedVal;
                    }
                    float forgetGate = sigmoid(forgetSum);
                    float inputGate = sigmoid(inputSum);
                    float candidateGate = sigmoid(candidateSum);
                    float outputGate = sigmoid(outputSum);

                    float newCell = (forgetGate * cellState[layer][j]) + (inputGate * candidateGate); // update the cell state
                    cellState[layer][j] = newCell;

                    hiddenState[layer][j] = outputGate * (float) Math.tanh(newCell);
                }
                currentLayerInput = hiddenState[layer]; // next input this hidden layer state
            }

            int lastHiddenSize = currentLayerInput.length;
            float[] output = new float[outputSize];
            for (int j = 0; j < outputSize; j++) {
                float sum = denseBias[j];
                int rowOffset = j * lastHiddenSize;

                for (int k = 0; k < lastHiddenSize; k++) {
                    sum += denseWeights[rowOffset + k] * currentLayerInput[k];
                }
                if (denseOutputActivations[j] == Activation.SIGMOID) {
                    output[j] = sigmoid(sum);
                }
            }

            System.arraycopy(output, 0, previousOutput, 0, outputSize);
            return output;
        }

    public void resetState() {
        for (int i = 0; i < lstmLayerDepth; i++) {
            Arrays.fill(hiddenState[i], 0.0f);
            Arrays.fill(cellState[i], 0.0f);
        }
        Arrays.fill(previousOutput, 0.0f);
    }
    }
