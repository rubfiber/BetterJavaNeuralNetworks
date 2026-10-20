package org.example.Usage.Convolutional;

import org.network.Convolutional.ConvolutionalNetwork;
import org.network.FeedForward.Activation;
import org.network.util;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class trainCNN {
    public static void main(String[] args) throws IOException {

        System.out.println("Building YOLO-style Convolutional Neural Network...");

        String[] fruits = new String[]{"apple", "Apple Granny Smith",
                "Apricot", "Avocado", "Banana", "Blueberry", "Cactus fruit",
                "Cantaloupe", "Cherry", "Clementine", "Corn", "cucumber", "Grape Blue",
                "Kiwi", "Lemon", "lime", "Mango", "Onion White", "Orange", "Papaya", "Passion Fruit",
                "Peach", "Pear", "Pepper Green", "Pepper Red", "Pineapple", "Plum", "Pomegranate",
                "potato", "Raspberry", "Strawberry", "Tomato", "Watermelon"}; // 33 fruits

        int[] cnnFilters = {16, 32, 64};
        int[] cnnInputChannels = {3, 16, 32};
        int[] cnnStrides = {2, 2, 2};

        int[] padding = {2, 2};
        int[][] dummyInputSize = new int[32][32];
        ConvolutionalNetwork.filterBuilder builder = new ConvolutionalNetwork.filterBuilder(dummyInputSize, padding, 32, 3);

        // --- YOLO MATH ---
        int gridSize = 4;
        int numCells = gridSize * gridSize; // 16 cells
        int featuresPerCell = 1 + 4 + fruits.length; // 1 Confidence + 4 Box (x,y,w,h) + 33 Classes = 38
        int totalOutputs = numCells * featuresPerCell; // 16 * 38 = 608 Output Neurons

        int[] ffnLayers = {2304, 256, totalOutputs};

        ConvolutionalNetwork cnn = new ConvolutionalNetwork(builder, ffnLayers, cnnFilters, cnnInputChannels, cnnStrides);

        cnn.network.fillOutputActivation(Activation.SIGMOID);

        // instead of add data I use add data grids since I am using yolo-style grid splitting
        cnn.addDataGrids("src/main/resources/train", fruits, 64, 0.2f, 0.3f, 0.4f, 0.4f, Color.WHITE);

        System.out.println("Beginning Training on " + cnn.outputs.length + " samples...");
        cnn.trainDataset(cnn.floatImages, cnn.outputs, 0.01f, 32, 10);

        System.out.println("Network trained successfully!");

        cnn.saveState("src/main/java/org/example/Convolutional/CNNState.bin", "src/main/java/org/example/Convolutional/FFNWeights.bin", "src/main/java/org/example/Convolutional/FFNBias.bin");

        long startTime = System.nanoTime();
        BufferedImage originalImage = ImageIO.read(new File("Data/Smith_2.jpg"));

        int targetSize = 64;
        BufferedImage image = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.drawImage(originalImage, 0, 0, targetSize, targetSize, null);
        g2d.dispose();

        float[][][] floatArray = new float[3][targetSize][targetSize];
        for (int y = 0; y < targetSize; y++) {
            for (int x = 0; x < targetSize; x++) {
                int rgb = image.getRGB(x, y);
                floatArray[0][y][x] = ((rgb >> 16) & 0xFF) / 255.0f;
                floatArray[1][y][x] = ((rgb >> 8) & 0xFF) / 255.0f;
                floatArray[2][y][x] = (rgb & 0xFF) / 255.0f;
            }
        }

        float[] prediction = cnn.predict(floatArray);
        long endTime = System.nanoTime();
        System.out.println("\nPrediction completed in " + (endTime - startTime) / 1_000_000 + " ms");

        // --- YOLO GRID DECODER ---
        float bestConfidence = 0.0f;
        int bestCellStartIdx = -1;
        int bestCellX = -1;
        int bestCellY = -1;

        // scan 16 cells for highest confidence
        for (int cellY = 0; cellY < gridSize; cellY++) {
            for (int cellX = 0; cellX < gridSize; cellX++) {
                int startIdx = ((cellY * gridSize) + cellX) * featuresPerCell;
                float confidence = prediction[startIdx];

                if (confidence > bestConfidence) {
                    bestConfidence = confidence;
                    bestCellStartIdx = startIdx;
                    bestCellX = cellX;
                    bestCellY = cellY;
                }
            }
        }

        if (bestConfidence > 0.5f) { // confidence threshhold: 50%
            float relX = prediction[bestCellStartIdx + 1];
            float relY = prediction[bestCellStartIdx + 2];
            float boxW = prediction[bestCellStartIdx + 3];
            float boxH = prediction[bestCellStartIdx + 4];

            // converting to normalized cell coordinates
            float finalNormalizedX = (bestCellX + relX) / gridSize;
            float finalNormalizedY = (bestCellY + relY) / gridSize;

            float[] classProbs = Arrays.copyOfRange(prediction, bestCellStartIdx + 5, bestCellStartIdx + featuresPerCell);
            int classIndex = util.getIndexOfMax(classProbs);

            System.out.println("Cell: [" + bestCellX + ", " + bestCellY + "]");
            System.out.println("Confidence: " + (bestConfidence * 100) + "%");
            System.out.println("Prediction: " + fruits[classIndex] + " " + (classProbs[classIndex] * 100) + "%");
            System.out.println("Position (Normalized): X=" + finalNormalizedX + ", Y=" + finalNormalizedY);
        } else {
            System.out.println("No objects detected. (Highest confidence was " + (bestConfidence * 100) + "%)");
        }
    }
}