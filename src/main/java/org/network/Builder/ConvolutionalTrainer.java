package org.network.Builder;

import org.network.Convolutional.ConvolutionalNetwork;
import org.network.FeedForward.Activation;
import java.awt.*;

public class ConvolutionalTrainer {
    String[] classes;
    int[] cnnFilters, cnnInputChannels, cnnStrides, padding, ffnLayers;
    int gridSize;
    String CNNStatePath, FFNWeightPath, FFNBiasPath;
    String dataPath;
    public ConvolutionalTrainer(String[] classes, int[] cnnFilters, int[] cnnInputChannels, int[] cnnStrides,
                                      int[] padding, int[] ffnLayers, int gridSize,
                                      String CNNStatePath, String FFNWeightPath, String FFNBiasPath,
                                String dataPath) {
        this.classes = classes;
        this.cnnFilters = cnnFilters;
        this.cnnInputChannels = cnnInputChannels;
        this.cnnStrides = cnnStrides;
        this.padding = padding;
        this.ffnLayers = ffnLayers;
        this.gridSize = gridSize;
        this.CNNStatePath = CNNStatePath;
        this.FFNWeightPath = FFNWeightPath;
        this.FFNBiasPath = FFNBiasPath;
        this.dataPath = dataPath;

    }
    void train() {

        System.out.println("Building YOLO-style Convolutional Neural Network...");


        int[][] dummyInputSize = new int[32][32];
        ConvolutionalNetwork.filterBuilder builder = new ConvolutionalNetwork.filterBuilder(dummyInputSize, padding, 32, 3);

        // --- YOLO MATH ---
        int gridSize = 4;
        int numCells = gridSize * gridSize; // 16 cells
        int featuresPerCell = 1 + 4 + classes.length; // 1 Confidence + 4 Box (x,y,w,h) + 33 Classes = 38
        int totalOutputs = numCells * featuresPerCell; // 16 * 38 = 608

        int[] ffnLayers = {2304, 256, totalOutputs};

        ConvolutionalNetwork cnn = new ConvolutionalNetwork(builder, ffnLayers, cnnFilters, cnnInputChannels, cnnStrides);

        cnn.network.fillOutputActivation(Activation.SIGMOID);
        cnn.network.featuresPerCell = featuresPerCell;

        // yolo
        cnn.addDataGrids(dataPath, classes, 64, 0.2f, 0.3f, 0.4f, false);

        //   cnn.saveSamples("Data/", classes, 10);

        System.out.println("Beginning Training on " + cnn.outputs.length + " samples...");
        cnn.trainDataset(cnn.floatImages, cnn.outputs, 0.01f, 32, 25);

        System.out.println("Network trained successfully!");

        cnn.saveState(CNNStatePath, FFNWeightPath, FFNBiasPath);

    }
}