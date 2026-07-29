package org.network.FeedForward;

import org.network.neuron.Neuron;

import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;

public class FeedForwardNetwork implements Network, Neuron{
    int inputSize;
    int hiddenLayerWidth;
    int hiddenLayerDepth;
    int outputSize;
    int getOutputSize;
    public float[][] hiddenLayerWeights;
    public float[][] hiddenLayerBias;
    public float[] outputLayerWeights;
    public float[] outputLayerBias;

    public FeedForwardNetwork(int inputSize, int hiddenLayerWidth, int hiddenLayerDepth, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayerWidth = hiddenLayerWidth;
        this.hiddenLayerDepth = hiddenLayerDepth;

    }

    @Override
    public float[] Train(float[] test, float[] desiredOutputs) {
        return new float[0];
    }

    @Override
    public float[] run(float[] input) {
        assert input.length == inputSize : "Input length has to be the same as input layer size";

        return new float[0];
    }

    @Override
    public void loadState(File file) {

    }

    @Override
    public void saveState(Path path) {

    }
//Neuron functions
    @Override
    public void initialize() {

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
