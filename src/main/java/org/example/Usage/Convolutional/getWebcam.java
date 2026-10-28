package org.example.Usage.Convolutional;

import com.github.sarxos.webcam.Webcam;

import java.util.List;
import java.util.NoSuchElementException;

public class getWebcam {
    public static Webcam getPreferredWebcam() throws NoSuchElementException {
        List<Webcam> webcams = Webcam.getWebcams();
        Webcam preferred = Webcam.getWebcamByName("C270 HD WEBCAM /dev/video2");
        Webcam defaultCam = Webcam.getDefault();
        webcams.add(preferred);
        webcams.add(defaultCam);
        for (Webcam cam : webcams) {
            try {
                if (cam != null) return cam;
            } catch (Exception ignored) {
            }
        }
        throw new NoSuchElementException("No webcams found");
    }
}
