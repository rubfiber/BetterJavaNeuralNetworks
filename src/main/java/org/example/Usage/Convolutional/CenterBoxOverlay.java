package org.example.Usage.Convolutional;

import javax.swing.JComponent;
import java.awt.*;

public class CenterBoxOverlay extends JComponent {
    private int boxX;
    private int boxY;

    private String predictionText = "";
    public CenterBoxOverlay(int boxX, int boxY) {
        this.boxX = boxY;
        this.boxY = boxY;
        setOpaque(false);
    }
    int centerX = 0;
    int centerY = 0;

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
             startX = (width - boxX) / 2;
             startY = (height - boxY) / 2;

        } else {
             startX =  centerX;
             startY = centerY;
        }

        // bounding box
        g2d.setColor(Color.GREEN);
        g2d.setStroke(new BasicStroke(3));
        g2d.drawRect(startX, startY, boxX, boxY);

        // crosshair
        g2d.setStroke(new BasicStroke(1));
        g2d.drawLine(width / 2 - 10, height / 2, width / 2 + 10, height / 2);
        g2d.drawLine(width / 2, height / 2 - 10, width / 2, height / 2 + 10);

        if (predictionText != null && !predictionText.isEmpty()) {
            g2d.setColor(Color.YELLOW); // High contrast color
            g2d.setFont(new Font("Arial", Font.BOLD, 24));

            g2d.setColor(Color.BLACK);
            g2d.drawString(predictionText, startX + 2, startY - 8);

            // draw text
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

    public void updateBox(int x, int y, int w, int h, String text) {
        this.centerX = x; this.centerY = y; this.boxX = w; this.boxY = h;
        this.predictionText = text;
        repaint();
    }
}