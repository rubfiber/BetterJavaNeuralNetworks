package org.example;

import org.network.FeedForward.FeedForwardNetwork;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
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
        network.train(trainingData, targetData, 100000, 0.1f);
        long endTime = System.nanoTime();

        long durationNs = endTime - startTime;
        double durationMs = durationNs / 1_000_000.0;


        System.out.println("Predictions after training:");
        for (float[] trainingDatum : trainingData) {
            System.out.println(Arrays.toString(network.predict(trainingDatum)));
        }
        System.out.println("Time in ms: " + durationMs);
    }
}