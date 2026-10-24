package org.example.Usage.Convolutional;

import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamPanel;
import org.jspecify.annotations.NonNull;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.Dimension;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class WebcamTrain {

    private static final String SAVE_DIRECTORY = "src/main/resources/captured_images";
    private static final int CROP_SIZE = 400;
    private static String FILE_NAME = "Strawberry";

    public static void main(String[] args) {
        File outputFolder = new File(SAVE_DIRECTORY);
        if (!outputFolder.exists()) {
            outputFolder.mkdirs();
        }


        Webcam webcam = Webcam.getWebcamByName("C270 HD WEBCAM /dev/video3"); //Logitech
        System.out.println("Showing live feed from webcam " + webcam.getDevice().getName());
        webcam.setViewSize(new Dimension(640, 480));
        webcam.open();

        JFrame window = new JFrame("Camera");

        // 1. Create WebCam Panel
        JLayeredPane layeredPane = getJLayeredPane(webcam);

        window.add(layeredPane);
        window.pack();
        window.setResizable(false);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);

        window.addKeyListener(new KeyAdapter() {
            private int captureCount = 0;

            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_SPACE) {
                    BufferedImage frame = webcam.getImage();

                    if (frame != null) {
                        int frameWidth = frame.getWidth();
                        int frameHeight = frame.getHeight();

                        int startX = (frameWidth - CROP_SIZE) / 2;
                        int startY = (frameHeight - CROP_SIZE) / 2;

                        BufferedImage croppedImage = frame.getSubimage(startX, startY, CROP_SIZE, CROP_SIZE);

                        try {
                            captureCount++;
                            String fileName = FILE_NAME + captureCount + "_" + System.currentTimeMillis() + ".png";
                            File outputFile = new File(outputFolder, fileName);
                            ImageIO.write(croppedImage, "png", outputFile);

                            System.out.println("Saved 100x100 center crop to: " + outputFile.getAbsolutePath());
                        } catch (IOException ex) {
                            System.err.println("Failed to save image: " + ex.getMessage());
                        }
                    }
                }
            }
        });

        // Request focus so key listener responds immediately without needing extra clicks
        window.requestFocusInWindow();
    }

    private static @NonNull JLayeredPane getJLayeredPane(Webcam webcam) {
        WebcamPanel panel = new WebcamPanel(webcam);
        panel.setFPSDisplayed(true);
        panel.setMirrored(true);
        panel.setBounds(0, 0, 640, 480);

        CenterBoxOverlay overlay = new CenterBoxOverlay(CROP_SIZE, CROP_SIZE);
        overlay.setBounds(0, 0, 640, 480);

        JLayeredPane layeredPane = new JLayeredPane();
        layeredPane.setPreferredSize(new Dimension(640, 480));
        layeredPane.add(panel, JLayeredPane.DEFAULT_LAYER); // Bottom layer
        layeredPane.add(overlay, JLayeredPane.PALETTE_LAYER); // Top layer
        return layeredPane;
    }
}