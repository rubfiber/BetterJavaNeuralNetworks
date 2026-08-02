package org.network.FeedForward;

import java.io.File;
import java.nio.file.Path;

public interface Network {

    void train(float[][] inputs, float[][] desiredOutputs, int epochs, float learningRate);
     float[] predict(float[] input);

     void loadState(File file);
     void saveState(Path path);

}
