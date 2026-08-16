package org.example.Usage.Convolutional;

import javax.swing.JComponent;
import java.awt.*;

public class CenterBoxOverlay extends JComponent {
    private final int boxSize;
    private String predictionText = "";
    public CenterBoxOverlay(int boxSize) {
        this.boxSize = boxSize;
        setOpaque(false);
    }
    int centerX = 0;
    int centerY = 0;
    public CenterBoxOverlay(int boxSize, int x, int y) {
        this.boxSize = boxSize;
        this.centerX = x;
        this.centerY = y;
        setOpaque(false);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        int width = getWidth();
        int height = getHeight();

        // center
        int startX;
        int startY;
        if (centerX == 0 && centerY == 0) {
             startX = (width - boxSize) / 2;
             startY = (height - boxSize) / 2;

        } else {
             startX =  centerX;
             startY = centerY;
        }

        // bounding box
        g2d.setColor(Color.GREEN);
        g2d.setStroke(new BasicStroke(3)); //3px thickness
        g2d.drawRect(startX, startY, boxSize, boxSize);

        // crosshair
        g2d.setStroke(new BasicStroke(1));
        g2d.drawLine(width / 2 - 10, height / 2, width / 2 + 10, height / 2);
        g2d.drawLine(width / 2, height / 2 - 10, width / 2, height / 2 + 10);

        if (predictionText != null && !predictionText.isEmpty()) {
            g2d.setColor(Color.YELLOW); // High contrast color
            g2d.setFont(new Font("Arial", Font.BOLD, 24));

            g2d.setColor(Color.BLACK);
            g2d.drawString(predictionText, startX + 2, startY - 8);

            // Draw the actual yellow text
            g2d.setColor(Color.YELLOW);
            g2d.drawString(predictionText, startX, startY - 10);
        }
    }
    public void updatePosition(int x, int y, String text) {
        this.centerX = x;
        this.centerY = y;
        this.predictionText = text;
        this.repaint();
    }
}