package org.example.Usage;

import org.network.Dataset;
import org.network.FeedForward.Activation;
import org.network.FeedForward.FeedForwardNetwork;

import java.io.IOException;
import java.util.Arrays;


public class Main {
    public static void main(String[] args) throws IOException {
        FeedForwardNetwork network = new FeedForwardNetwork(new int[]{2, 3, 4, 1});
        network.fillOutputActivation(Activation.RELU);

        // XOR
        float[][] trainingData = {
                {1.0f, 0.0f},
                {0.0f, 0.0f},
                {1.0f, 1.0f},
                {0.0f, 1.0f}
        };

        //-1 false, 1 true
        float[][] targetData = {
                { 1.0f},
                { 0.0f},
                { 0.0f},
                { 1.0f}
        };
        System.out.println("XOR based on inputs:" +  Arrays.deepToString(trainingData));
        System.out.println("Predictions before training:");
        for (float[] datum : trainingData) {
            System.out.println(Arrays.toString(network.predict(datum)));
        }

        long startTime = System.nanoTime();
        network.trainDataset(new Dataset(trainingData, targetData), 100000, 1, 0.01f);
        long endTime = System.nanoTime();
        network.saveState("src/main/java/org/example/exampleWeights.bin", "src/main/java/org/example/exampleBias.bin");
        long durationNs = endTime - startTime;
        double durationMs = durationNs / 1_000_000.0;

        network.loadWeights("src/main/java/org/example/exampleWeights.bin");
       network.loadBias("src/main/java/org/example/exampleBias.bin");
        System.out.println("Predictions after training:");
        for (float[] trainingDatum : trainingData) {
            System.out.println(Arrays.toString(network.predict(trainingDatum)));
        }
        System.out.println("Training time in ms: " + durationMs);
    }
}