package org.example.Usage;

import org.network.Convolutional.ConvolutionalNetwork;
import org.network.util;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class MainConvolutional {
     static void main(String[] args) throws IOException {

        System.out.println("Building Convolutional Neural Network...");

        //test values - it will identify fruits
        String[] fruits = new String[]{"Apple Braeburn", "Apple Granny Smith",
                "Apricot", "Avocado", "Banana", "Blueberry", "Cactus fruit",
                "Cantaloupe", "Cherry", "Clementine", "Corn", "Cucumber Ripe", "Grape Blue",
                "Kiwi", "Lemon", "Limes", "Mango", "Onion White", "Orange", "Papaya", "Passion Fruit",
                "Peach", "Pear", "Pepper Green", "Pepper Red", "Pineapple", "Plum", "Pomegranate",
                "Potato Red", "Raspberry", "Strawberry", "Tomato", "Watermelon"};

        int[] cnnFilters = {16, 32, 64};           // How many filters each layer outputs
        int[] cnnInputChannels = {3, 16, 32};      // How many channels each layer takes in
        int[] cnnStrides = {2, 2, 2};            // How many pixels the filter jumps

        int[] padding = {2, 2};

        int[][] dummyInputSize = new int[32][32];
        ConvolutionalNetwork.filterBuilder builder = new ConvolutionalNetwork.filterBuilder(dummyInputSize, padding, 32, 3);

        int[] ffnLayers = {2304, 256, 33}; //num fruits is 33

        ConvolutionalNetwork cnn = new ConvolutionalNetwork(builder, ffnLayers, cnnFilters, cnnInputChannels, cnnStrides);
       // cnn.addData("Data/TrainFruits/", fruits, 64);

     //   cnn.trainDataset(cnn.floatImages, cnn.outputs, 0.01f, 32, 10);

        System.out.println("Network built successfully!");

        long startTime = System.nanoTime();

        BufferedImage originalImage = ImageIO.read(new File("Data/Smith_2.jpg"));

        // 64 x 64
        int targetSize = 64;
        BufferedImage image = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g2d = image.createGraphics();
        g2d.drawImage(originalImage, 0, 0, targetSize, targetSize, null);
        g2d.dispose();

        int channels = 3;
        float[][][] floatArray = new float[channels][image.getHeight()][image.getWidth()];

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);

                // RGB bitwise
                float r = ((rgb >> 16) & 0xFF);
                float g = ((rgb >> 8) & 0xFF);
                float b = (rgb & 0xFF);

                floatArray[0][y][x] = r;
                floatArray[1][y][x] = g;
                floatArray[2][y][x] = b;
            }
        }
        System.out.println("Saving and loading..."); //test save and load
     //   cnn.saveState("src/main/java/org/example/Convolutional/CNNState.bin", "src/main/java/org/example/Convolutional/FFNWeights.bin", "src/main/java/org/example/Convolutional/FFNBias.bin");
        cnn.loadState("src/main/java/org/example/Convolutional/CNNState.bin", "src/main/java/org/example/Convolutional/FFNWeights.bin", "src/main/java/org/example/Convolutional/FFNBias.bin");

         float[] prediction = cnn.predict(floatArray);

        long endTime = System.nanoTime();

        System.out.println("\nPrediction completed in " + (endTime - startTime) + " ns");
        System.out.println("Output (Softmax Probabilities): " + Arrays.toString(prediction));

        int max = util.getIndexOfMax(prediction);
        System.out.println("Fruit: " + fruits[max]);
    }
}