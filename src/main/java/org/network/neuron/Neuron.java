package org.network.neuron;

public interface Neuron {
    public void initializeHiddenLayers();
    public void initializeOutputLayer();
    public float neuronCompute(float[] weights, float bias, float[] input, boolean isOutLayer);

    public float getWeights(int neuronIdx);
    public float getBias(int neuronIdx);

    public void setWeights(int neuronIdx, float value);
    public void setBias(int neuronIdx, float value);

}
