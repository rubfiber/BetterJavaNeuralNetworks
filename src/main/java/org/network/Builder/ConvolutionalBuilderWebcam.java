package org.network.Builder;

import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamPanel;
import org.example.Usage.Convolutional.CenterBoxOverlay;
import org.network.Convolutional.ConvolutionalNetwork;
import org.network.FeedForward.Activation;
import org.network.util.AppFiles;
import org.network.util.util;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Arrays;
import java.util.NoSuchElementException;

import static org.example.Usage.Convolutional.getWebcam.getPreferredWebcam;

public class ConvolutionalBuilderWebcam {
    String[] classes;
    int[] cnnFilters, cnnInputChannels, cnnStrides, padding, ffnLayers;
    int gridSize;
    String CNNStatePath, FFNWeightPath, FFNBiasPath;
    public ConvolutionalBuilderWebcam(String[] classes, int[] cnnFilters, int[] cnnInputChannels, int[] cnnStrides,
                                      int[] padding, int[] ffnLayers, int gridSize,
                                      String CNNStatePath, String FFNWeightPath, String FFNBiasPath) {
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

    }
      void run() throws NoSuchElementException, IOException {

        int[][] dummyInputSize = new int[32][32];
        ConvolutionalNetwork.filterBuilder builder = new ConvolutionalNetwork.filterBuilder(dummyInputSize, padding, 32, 3);
        int numCells = gridSize * gridSize; // 16
        int featuresPerCell = 1 + 4 + classes.length; // 38
        int totalOutputs = numCells * featuresPerCell; // 608

        int[] ffnLayers = {2304, 256, totalOutputs};
        ConvolutionalNetwork cnn = new ConvolutionalNetwork(builder, ffnLayers, cnnFilters, cnnInputChannels, cnnStrides);

        cnn.network.fillOutputActivation(Activation.SIGMOID);
        cnn.loadState(AppFiles.resourceToTempFile(CNNStatePath),
                AppFiles.resourceToTempFile(FFNWeightPath),
                AppFiles.resourceToTempFile(FFNBiasPath)); // because it might break when using a jar
        int channels = 3;

        Webcam webcam =  getPreferredWebcam();
        webcam.open();
        System.out.println("Reading live RGB data. Press Ctrl+C to stop.");

        WebcamPanel panel = new WebcamPanel(webcam);
        panel.setFPSDisplayed(true);
        panel.setMirrored(true);

        panel.setBounds(0, 0, 640, 480);

        JFrame window = new JFrame("CNN");

        window.setResizable(false);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        CenterBoxOverlay overlay = new CenterBoxOverlay(400, 400);
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
                    System.out.println("Prediction: " + Arrays.toString(prediction));

                    System.out.println("\nPrediction completed in " + (endTime - startTime) / 1_000_000 + " ms");

                    float bestConf = 0f;
                    int bestIdx = -1, bestCX = 0, bestCY = 0;
                    for (int cellY = 0; cellY < gridSize; cellY++) {
                        for (int cellX = 0; cellX < gridSize; cellX++) {
                            int startIdx = ((cellY * gridSize) + cellX) * featuresPerCell;
                            if (prediction[startIdx] > bestConf) {
                                bestConf = prediction[startIdx];
                                bestIdx = startIdx;
                                bestCX = cellX;
                                bestCY = cellY;
                            }
                        }
                    }

                    if (bestConf > 0.25f) {
                        float[] classProbs = Arrays.copyOfRange(prediction, bestIdx + 5, bestIdx + 5 + classes.length);
                        int max = util.getIndexOfMax(classProbs);

                        int pixelX = 640 - (int) (((bestCX + prediction[bestIdx + 1]) / gridSize) * 640); // un-mirror
                        int pixelY = (int) (((bestCY + prediction[bestIdx + 2]) / gridSize) * 480);

                        String label = classes[max] + " " + Math.round(classProbs[max] * 1000) / 10.0 + "% (conf "
                                + Math.round(bestConf * 100) + "%)";
                        int bw = Math.round(prediction[bestIdx + 3] * 640), bh = Math.round(prediction[bestIdx + 4] * 480);
                        //best cell
                        SwingUtilities.invokeLater(() -> overlay.updateBox(pixelX - bw / 2, pixelY - bh / 2, bw, bh, label));
                    } else {
                        SwingUtilities.invokeLater(() -> overlay.updatePosition(-1000, -1000, ""));
                    }
                }

                try { Thread.sleep(33); } catch (InterruptedException e) { break; }
            }
        }).start();
    }
}

