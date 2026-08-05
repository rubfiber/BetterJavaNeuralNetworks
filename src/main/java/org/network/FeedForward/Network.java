package org.network.FeedForward;

import java.io.File;
import java.nio.file.Path;

public interface Network {

    void trainDataset(float[][] inputs, float[][] desiredOutputs, int epochs, int batchSize, float learningRate);
     float[] predict(float[] input);
    static float[] acceleratedPrediction(float[] input) {
        return input;
    }


}
