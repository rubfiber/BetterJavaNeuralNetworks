package org.example;

import org.network.FeedForward.FeedForwardNetwork;

import java.io.IOException;
import java.util.Arrays;

import static org.network.FeedForward.FeedForwardNetwork.saveState;

public class Main {
    public static void main(String[] args) throws IOException {
        FeedForwardNetwork network = new FeedForwardNetwork(2, 4, 2, 1);

        // XOR
        float[][] trainingData = {
                {0.0f, 0.0f},
                {0.0f, 1.0f},
                {1.0f, 0.0f},
                {1.0f, 1.0f}
        };

        //-1 false, 1 true
        float[][] targetData = {
                {-1.0f},
                { 1.0f},
                { 1.0f},
                {-1.0f}
        };
        System.out.println("XOR based on inputs:" +  Arrays.deepToString(trainingData));
        System.out.println("Predictions before training:");
        for (float[] datum : trainingData) {
            System.out.println(Arrays.toString(network.predict(datum)));
        }

        long startTime = System.nanoTime();
        network.trainDataset(trainingData, targetData, 100000, 1, 0.01f);
        long endTime = System.nanoTime();
        saveState(network.hiddenLayerWeights, network.hiddenLayerBias, network.outputLayerWeights, network.outputLayerBias, "src/main/java/org/example/exampleHiddenWeights.bin", "src/main/java/org/example/exampleHiddenBias.bin", "src/main/java/org/example/exampleOutputWeights.bin", "src/main/java/org/example/exampleOutputBias.bin");
        long durationNs = endTime - startTime;
        double durationMs = durationNs / 1_000_000.0;

        network.loadWeights("src/main/java/org/example/exampleHiddenWeights.bin", "src/main/java/org/example/exampleOutputWeights.bin");
       network.loadBias("src/main/java/org/example/exampleHiddenBias.bin", "src/main/java/org/example/exampleOutputBias.bin");
        System.out.println("Predictions after training:");
        for (float[] trainingDatum : trainingData) {
            System.out.println(Arrays.toString(network.predict(trainingDatum)));
        }
        System.out.println("Training time in ms: " + durationMs);
    }
}