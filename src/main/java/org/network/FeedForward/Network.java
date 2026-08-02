package org.network.FeedForward;

import java.io.File;
import java.nio.file.Path;

public interface Network {

    public void train(float[][] inputs, float[][] desiredOutputs, int epochs, float learningRate);
    public float[] predict(float[] input);

    public void loadState(File file);
    public void saveState(Path path);

}
