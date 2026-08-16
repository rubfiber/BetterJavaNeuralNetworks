package org.example.Usage.Convolutional;

import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamPanel;
import com.github.sarxos.webcam.WebcamResolution;
import org.network.Convolutional.ConvolutionalNetwork;
import org.network.FeedForward.Activation;
import org.network.util;

import javax.imageio.ImageIO;
import javax.swing.*;
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

        int[] ffnLayers = {2304, 256, 35}; //num fruits is 33 + (x,y) pos

        ConvolutionalNetwork cnn = new ConvolutionalNetwork(builder, ffnLayers, cnnFilters, cnnInputChannels, cnnStrides);

        cnn.network.setOutputActivationWithIndex(cnn.network.outputSize - 1, Activation.SIGMOID);
        cnn.network.setOutputActivationWithIndex(cnn.network.outputSize - 2, Activation.SIGMOID);
        cnn.loadState("src/main/java/org/example/Convolutional/CNNState.bin", "src/main/java/org/example/Convolutional/FFNWeights.bin", "src/main/java/org/example/Convolutional/FFNBias.bin");

        int channels = 3;

        Webcam webcam = Webcam.getWebcamByName("C270 HD WEBCAM /dev/video3"); //Logitech
        webcam.setViewSize(new Dimension(640, 480));
        webcam.open();

        System.out.println("Reading live RGB data. Press Ctrl+C to stop.");


        WebcamPanel panel = new WebcamPanel(webcam);
        panel.setFPSDisplayed(true);
        panel.setMirrored(true);

        panel.setBounds(0, 0, 640, 480);

        JFrame window = new JFrame("Java Webcam Viewer");
        window.setResizable(false);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        CenterBoxOverlay overlay = new CenterBoxOverlay(400, 0, 0);
        overlay.setBounds(0, 0, 640, 480);

        JLayeredPane layeredPane = new JLayeredPane();
        layeredPane.setPreferredSize(new Dimension(640, 480));
        layeredPane.add(panel, JLayeredPane.DEFAULT_LAYER);
        layeredPane.add(overlay, JLayeredPane.PALETTE_LAYER);

        window.add(layeredPane);
        window.pack();
        window.setVisible(true);
        int targetSize = 64;
        new Thread(() -> {

            while (webcam.isOpen()) {
                BufferedImage frame = webcam.getImage(); // raw frame


                if (frame != null) {
                    float[][][] floatArray = new float[channels][targetSize][targetSize];

                    BufferedImage image = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
                    java.awt.Graphics2D g2d = image.createGraphics();

                    g2d.drawImage(frame, 0, 0, targetSize, targetSize, null);
                    g2d.dispose();

                    for (int y = 0; y < targetSize; y++) {
                        for (int x = 0; x < targetSize; x++) {

                            int pixel = image.getRGB(x, y);

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

                    int max = util.getIndexOfMax(Arrays.copyOfRange(prediction, 0, fruits.length));
                    System.out.println("Fruit: " + fruits[max]);
                    System.out.println("Output: " + Arrays.toString(prediction));
                    float[] pos = (Arrays.copyOfRange(prediction, fruits.length, prediction.length));
                    System.out.println("Normalized pos: " + Arrays.toString(pos));

                    int pixelX = (int) (pos[0] * 640);
                    int pixelY = (int) (pos[1] * 480);

                    overlay.updatePosition(pixelX, pixelY, fruits[max] + " " + Math.round(prediction[max] * 100 * 10.0) / 10.0 + "%");
                }

                try { Thread.sleep(33); } catch (InterruptedException e) { break; }
            }
        }).start();
    }
}
