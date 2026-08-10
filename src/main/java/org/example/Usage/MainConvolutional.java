package org.example.Usage;

import org.network.Convolutional.ConvolutionalNetwork;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class MainConvolutional {
     static void main(String[] args) throws IOException {

        System.out.println("Building Convolutional Neural Network...");

        int[] cnnFilters = {16, 32, 64};           // How many filters each layer outputs
        int[] cnnInputChannels = {3, 16, 32};      // How many channels each layer takes in
        int[] cnnStrides = {2, 2, 2};            // How many pixels the filter jumps

        int[] padding = {2, 2};

        int[][] dummyInputSize = new int[32][32];
        ConvolutionalNetwork.filterBuilder builder = new ConvolutionalNetwork.filterBuilder(dummyInputSize, padding, 32, 3);

        int[] ffnLayers = {1024, 64, 2};

        ConvolutionalNetwork cnn = new ConvolutionalNetwork(builder, ffnLayers, cnnFilters, cnnInputChannels, cnnStrides);
        cnn.addData("Data/Train/");

        cnn.trainDataset(cnn.floatImages, cnn.outputs, 0.01f, 32, 10);

        System.out.println("Network built successfully!");

        long startTime = System.nanoTime();

        BufferedImage originalImage = ImageIO.read(new File("Data/0041.jpg"));

// 2. Resize it to 32x32 to match the network's trained architecture
        int targetSize = 32;
        BufferedImage image = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g2d = image.createGraphics();
        g2d.drawImage(originalImage, 0, 0, targetSize, targetSize, null);
        g2d.dispose();

// 3. Now proceed to convert the correctly sized 'image' into a float array
        int channels = 3;
        float[][][] floatArray = new float[channels][image.getHeight()][image.getWidth()];

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);

                // Extract RGB components (0 - 255)
                // Optionally divide by 255.0f to normalize between 0.0 and 1.0
                float r = ((rgb >> 16) & 0xFF);
                float g = ((rgb >> 8) & 0xFF);
                float b = (rgb & 0xFF);

                floatArray[0][y][x] = r; // Red channel
                floatArray[1][y][x] = g; // Green channel
                floatArray[2][y][x] = b; // Blue channel
            }
        }

        float[] prediction = cnn.predict(floatArray);

        long endTime = System.nanoTime();

        System.out.println("\nPrediction completed in " + (endTime - startTime) + " ns");
        System.out.println("Output (Softmax Probabilities): " + Arrays.toString(prediction));
    }
}