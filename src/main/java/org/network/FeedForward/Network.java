package org.network.FeedForward;

import java.io.File;
import java.nio.file.Path;

public interface Network {

    public float[] train(float[] test, float[] desiredOutputs);
    public float[] predict(float[] input);

    public void loadState(File file);
    public void saveState(Path path);

}
