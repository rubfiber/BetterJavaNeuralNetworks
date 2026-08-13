package org.example.Usage.Convolutional;

import javax.swing.JComponent;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.BasicStroke;

public class CenterBoxOverlay extends JComponent {
    private final int boxSize;

    public CenterBoxOverlay(int boxSize) {
        this.boxSize = boxSize;
        setOpaque(false);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        int width = getWidth();
        int height = getHeight();

        int startX = (width - boxSize) / 2;
        int startY = (height - boxSize) / 2;

        //bounding box
        g2d.setColor(Color.GREEN);
        g2d.setStroke(new BasicStroke(3)); //3px thickness
        g2d.drawRect(startX, startY, boxSize, boxSize);

        // crosshair
        g2d.setStroke(new BasicStroke(1));
        g2d.drawLine(width / 2 - 10, height / 2, width / 2 + 10, height / 2);
        g2d.drawLine(width / 2, height / 2 - 10, width / 2, height / 2 + 10);
    }
}