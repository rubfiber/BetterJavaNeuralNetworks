package org.example.Usage.Convolutional;

import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamPanel;
import com.github.sarxos.webcam.WebcamResolution;
import org.network.Convolutional.ConvolutionalNetwork;
import org.network.util;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class WebcamTest {
    public static void main(String[] args) throws IOException {
        String[] fruits = new String[]{"apple", "Apple Granny Smith",
                "Apricot", "Avocado", "Banana", "Blueberry", "Cactus fruit",
                "Cantaloupe", "Cherry", "Clementine", "Corn", "cucumber", "Grape Blue",
                "Kiwi", "Lemon", "lime", "Mango", "Onion White", "Orange", "Papaya", "Passion Fruit",
                "Peach", "Pear", "Pepper Green", "Pepper Red", "Pineapple", "Plum", "Pomegranate",
                "potato", "Raspberry", "Strawberry", "Tomato", "Watermelon"};

        int[] cnnFilters = {16, 32, 64};           // How many filters each layer outputs
        int[] cnnInputChannels = {3, 16, 32};      // How many channels each layer takes in
        int[] cnnStrides = {2, 2, 2};            // How many pixels the filter jumps

        int[] padding = {2, 2};

        int[][] dummyInputSize = new int[32][32];
        ConvolutionalNetwork.filterBuilder builder = new ConvolutionalNetwork.filterBuilder(dummyInputSize, padding, 32, 3);

        int[] ffnLayers = {2304, 256, 33}; //num fruits is 33

        ConvolutionalNetwork cnn = new ConvolutionalNetwork(builder, ffnLayers, cnnFilters, cnnInputChannels, cnnStrides);
        cnn.loadState("src/main/java/org/example/Convolutional/CNNState.bin", "src/main/java/org/example/Convolutional/FFNWeights.bin", "src/main/java/org/example/Convolutional/FFNBias.bin");

        int channels = 3;

        Webcam webcam = Webcam.getDefault();
        webcam.setViewSize(new Dimension(640, 480));
        webcam.open();

        System.out.println("Reading live RGB data. Press Ctrl+C to stop.");


        WebcamPanel panel = new WebcamPanel(webcam);
        panel.setFPSDisplayed(true); // Shows framerate info
        panel.setMirrored(true);     // Mirrors video for standard view

        // Set up the window frame container
        JFrame window = new JFrame("Java Webcam Viewer");
        window.add(panel);
        window.setResizable(false);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.pack();
        window.setVisible(true);
        int targetSize = 64;
        new Thread(() -> {

            while (webcam.isOpen()) {
                BufferedImage frame = webcam.getImage(); // raw frame

                if (frame != null) {
                    // FIX 1: Initialize the array with targetSize, NOT frame height/width
                    float[][][] floatArray = new float[channels][targetSize][targetSize];

                    BufferedImage image = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
                    java.awt.Graphics2D g2d = image.createGraphics();

                    // FIX 2: Draw the 'frame' onto the 'image', not image onto image
                    g2d.drawImage(frame, 0, 0, targetSize, targetSize, null);
                    g2d.dispose();

                    for (int y = 0; y < targetSize; y++) {
                        for (int x = 0; x < targetSize; x++) {

                            int pixel = image.getRGB(x, y);

                            // FIX 3: Divide by 255.0f to match your training normalization!
                            float r = ((pixel >> 16) & 0xFF) / 255.0f;
                            float g = ((pixel >> 8) & 0xFF) / 255.0f;
                            float b = (pixel & 0xFF) / 255.0f;

                            floatArray[0][y][x] = r;
                            floatArray[1][y][x] = g;
                            floatArray[2][y][x] = b;
                        }
                    }

                    long startTime = System.nanoTime();
                    float[] prediction = cnn.predict(floatArray);
                    long endTime = System.nanoTime();

                    System.out.println("\nPrediction completed in " + (endTime - startTime) / 1_000_000 + " ms");
                    System.out.println("Output (Softmax Probabilities): " + Arrays.toString(prediction));

                    int max = util.getIndexOfMax(prediction);
                    System.out.println("Fruit: " + fruits[max]);
                }

                try { Thread.sleep(33); } catch (InterruptedException e) { break; }
            }
        }).start();
    }
}
