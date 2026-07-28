package org.network.FeedForward;

import java.io.File;
import java.nio.file.Path;

public interface Network {
    public double[] Train(double[] test, double[] desiredOutputs);
    public double[] run(double[] input);

    public void loadState(File file);
    public void saveState(Path path);

}
