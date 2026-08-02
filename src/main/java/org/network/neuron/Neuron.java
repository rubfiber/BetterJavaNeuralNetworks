package org.network.neuron;

public interface Neuron {
     void initializeHiddenLayers();
     void initializeOutputLayer();

     float getWeights(int neuronIdx);
     float getBias(int neuronIdx);

     void setWeights(int neuronIdx, float value);
     void setBias(int neuronIdx, float value);

}
