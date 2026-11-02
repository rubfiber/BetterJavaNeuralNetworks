package org.network.Recurrent;

import org.network.FeedForward.Activation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
    public float[][] recurrentGradients;
    public float[][] biasGradients;
    public Activation[] outputActivations;
    public Activation[][] hiddenActivations;

    public float[][] recurrentWeights;

    public final List<StepCache> history = new ArrayList<>();
    private boolean recording = false;

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
        recurrentGradients = new float[hiddenLayerDepth][];
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
            Arrays.fill(hiddenActivations[i],  Activation.TANH);

            recurrentWeights[i] = new float[nextLayerSize * nextLayerSize];
            recurrentGradients[i] = new float[nextLayerSize * nextLayerSize];
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

        if (recording) {
            float[][] snap = new float[hiddenLayerDepth][];
            for (int l = 0; l < hiddenLayerDepth; l++) snap[l] = hiddenLayerOutputs[l].clone();
            history.add(new StepCache(input.clone(), snap));
        }
        return previousOutput;
    }

    public void resetState() {
        for (float[] s : recurrentState) Arrays.fill(s, 0f);
        Arrays.fill(previousOutput, 0f);
        history.clear();
    }

    public float[][] forwardSequence(float[][] inputs) {
        resetState();
        recording = true;
        float[][] outs = new float[inputs.length][];
        for (int t = 0; t < inputs.length; t++) outs[t] = predict(inputs[t]);
        recording = false;
        return outs;
    }

    public void zeroGradients() {
        for (int l = 0; l < hiddenLayerDepth; l++) {
            Arrays.fill(weightGradients[l], 0f);
            Arrays.fill(biasGradients[l], 0f);
            Arrays.fill(recurrentGradients[l], 0f);
        }
    }
    public static class StepCache {
        public final float[] input;      // x[t]
        public final float[][] outputs;  // h[t] for every layer (copied)
        StepCache(float[] input, float[][] outputs) {
            this.input = input;
            this.outputs = outputs;
        }
    }
    public float backward(float[][] targets) {
        final int totalTime = history.size();
        final float EPS = 1e-7f;

        float[][] carry = new float[hiddenLayerDepth][];
        for (int l = 0; l < hiddenLayerDepth; l++) carry[l] = new float[layers[l + 1]];

        float loss = 0f;
        int supervised = 0;

        for (int t = totalTime - 1; t >= 0; t--) { // going backwards
            StepCache c = history.get(t);
            float[] dAbove = null; // gradient comes from layer above
            for (int layer = hiddenLayerDepth-1; layer >= 0; layer--) {
                final int inS = layers[layer];
                final int outS = layers[layer + 1];
                final boolean isOutput = (layer == hiddenLayerDepth - 1);

                float[] out = c.outputs[layer];
                float[] in = (layer == 0) ? c.input : c.outputs[layer - 1];
                float[] prev = (t > 0) ? history.get(t - 1).outputs[layer] : null; // null means zeros at t=0

                float[] delta = new float[outS];
                if (isOutput) {
                    if (targets[t] != null) {
                        supervised++;
                        for (int o = 0; o < outS; o++) {
                            float p = out[o];
                            float y = targets[t][o];
                            delta[o] = p - y;
                            loss -= y * (float) Math.log(p + EPS) + (1f - y) * (float) Math.log(1f - p + EPS);
                        }
                    }
                } else {
                    for (int o = 0; o < outS; o++) {
                        assert dAbove != null;
                        delta[o] = (dAbove[o] + carry[layer][o]) * (1f - out[o] * out[o]); // tanh'
                    }
                }

                float[] weightGrad = weightGradients[layer];
                float[] biasGrad = biasGradients[layer];
                float[] recurrentGrad = recurrentGradients[layer];
                for (int o = 0; o < outS; o++) {
                    float d = delta[o];
                    biasGrad[o] += d;
                    int wBase = o * inS;
                    for (int i = 0; i < inS; i++) weightGrad[wBase + i] += d * in[i];
                    if (!isOutput && prev != null) {
                        int rBase = o * outS;
                        for (int k = 0; k < outS; k++) recurrentGrad[rBase + k] += d * prev[k];
                    }
                }
                float[] dBelow = null;
                if (layer > 0) {
                    dBelow = new float[inS];
                    float[] w = hiddenLayerWeights[layer];
                    for (int o = 0; o < outS; o++) {
                        float d = delta[o];
                        int wBase = o * inS;
                        for (int i = 0; i < inS; i++) dBelow[i] += d * w[wBase + i];
                    }
                }

                if (!isOutput) {
                    float[] newCarry = new float[outS];
                    float[] rw = recurrentWeights[layer];
                    for (int o = 0; o < outS; o++) {
                        float d = delta[o];
                        int rBase = o * outS;
                        for (int k = 0; k < outS; k++) newCarry[k] += d * rw[rBase + k];
                    }
                    carry[layer] = newCarry;
                }

                dAbove = dBelow;
            }
        }

        if (supervised > 0) {
            float s = 1f / supervised;
            for (int l = 0; l < hiddenLayerDepth; l++) {
                for (int i = 0; i < weightGradients[l].length; i++) weightGradients[l][i] *= s;
                for (int i = 0; i < biasGradients[l].length; i++) biasGradients[l][i] *= s;
                for (int i = 0; i < recurrentGradients[l].length; i++) recurrentGradients[l][i] *= s;
            }
            loss *= s;
        }
        return loss;

        }
    public void clipGradients(float maxNorm) {
        double sq = 0;
        for (int l = 0; l < hiddenLayerDepth; l++) {
            for (float g : weightGradients[l]) sq += g * g;
            for (float g : biasGradients[l]) sq += g * g;
            if (l < hiddenLayerDepth - 1) for (float g : recurrentGradients[l]) sq += g * g; // skip unused output recurrence
        }
        double norm = Math.sqrt(sq);
        if (norm <= maxNorm) return;
        float s = (float) (maxNorm / norm);
        for (int l = 0; l < hiddenLayerDepth; l++) {
            for (int i = 0; i < weightGradients[l].length; i++) weightGradients[l][i] *= s;
            for (int i = 0; i < biasGradients[l].length; i++) biasGradients[l][i] *= s;
            for (int i = 0; i < recurrentGradients[l].length; i++) recurrentGradients[l][i] *= s;
        }
    }

    public void sgdStep(float lr) {
        for (int l = 0; l < hiddenLayerDepth; l++) {
            for (int i = 0; i < hiddenLayerWeights[l].length; i++) hiddenLayerWeights[l][i] -= lr * weightGradients[l][i];
            for (int i = 0; i < hiddenLayerBias[l].length; i++) hiddenLayerBias[l][i] -= lr * biasGradients[l][i];
            if (l < hiddenLayerDepth - 1)
                for (int i = 0; i < recurrentWeights[l].length; i++) recurrentWeights[l][i] -= lr * recurrentGradients[l][i];
        }
    }

    public float trainSequence(float[][] inputs, float[][] targets, float lr, float maxNorm) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException(
                    "Input sequence length (" + inputs.length + ") must match target sequence length (" + targets.length + ")"
            );
        }
            zeroGradients();
            forwardSequence(inputs);
            float loss = backward(targets);
            if (maxNorm > 0f) {
                clipGradients(maxNorm);
            }
            sgdStep(lr);
            return loss;
        }

    public void train(float[][][] datasetInput, float[][][] datasetTarget, int epochs, float learningRate, float maxNorm) {
        if (datasetInput.length != datasetTarget.length) {
            throw new IllegalArgumentException("input dataset size has to match target dataset size");
        }

        int numSequences = datasetInput.length;
        int[] indices = new int[numSequences];
        for (int i = 0; i < numSequences; i++) indices[i] = i;
        Random rng = new Random();

        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = numSequences - 1; i > 0; i--) {
                int swapIdx = rng.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[swapIdx];
                indices[swapIdx] = temp;
            }

            float epochLoss = 0f;
            for (int idx : indices) {
                epochLoss += trainSequence(datasetInput[idx], datasetTarget[idx], learningRate, maxNorm);
            }

            epochLoss /= numSequences;

            int logInterval = Math.max(1, epochs / 10);
            if (epoch % logInterval == 0 || epoch == 1 || epoch == epochs) {
                System.out.printf("Epoch %d/%d | Avg Loss: %.6f%n", epoch, epochs, epochLoss);
            }
        }
    }
}

