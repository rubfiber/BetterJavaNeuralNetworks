package org.example.Usage.Convolutional;

import org.network.Builder.ConvolutionalBuilderWebcam;
import org.network.Builder.ConvolutionalTrainer;

import java.io.IOException;

public class BuilderExample {
    int[] cnnFilters = {16, 32, 64};
    int[] cnnInputChannels = {3, 16, 32};
    int[] cnnStrides = {2, 2, 2};

    int[] padding = {2, 2};
    int gridSize = 4;
    String[] classes = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};

    int[] ffnLayers = {2304, 256, (1 + 4 + classes.length) * gridSize};

    public ConvolutionalTrainer trainer = new ConvolutionalTrainer(classes, cnnFilters, cnnInputChannels, cnnStrides, padding, ffnLayers, gridSize,
            "src/main/java/org/example/Convolutional/CNNNumbersState.bin", "src/main/java/org/example/Convolutional/FFNWeights.bin", "src/main/java/org/example/Convolutional/FFNBias.bin",
            "src/main/resources/archive/trainingSet/trainingSet/");

    public ConvolutionalBuilderWebcam builderWebcam = new ConvolutionalBuilderWebcam(classes, cnnFilters, cnnInputChannels, cnnStrides, padding, ffnLayers, gridSize, "src/main/java/org/example/Convolutional/CNNNumbersState.bin", "src/main/java/org/example/Convolutional/FFNWeights.bin", "src/main/java/org/example/Convolutional/FFNBias.bin");
    void main() throws IOException {
        trainer.train();

      //  builderWebcam.run();

    }
}
