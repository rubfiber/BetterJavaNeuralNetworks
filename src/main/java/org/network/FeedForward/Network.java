package org.network.FeedForward;

import org.network.Dataset;

import java.io.File;
import java.nio.file.Path;

public interface Network {

    void trainDataset(Dataset dataset, int epochs, int batchSize, float learningRate);
     float[] predict(float[] input);
    static float[] acceleratedPrediction(float[] input) {
        return input;
    }


}
