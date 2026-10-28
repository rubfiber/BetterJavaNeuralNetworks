package org.network;

import org.network.util.Dataset;

public interface Network {

    void trainDataset(Dataset dataset, int epochs, int batchSize, float learningRate);
     float[] predict(float[] input);


}
