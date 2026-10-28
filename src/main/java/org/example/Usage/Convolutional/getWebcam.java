package org.example.Usage.Convolutional;

import com.github.sarxos.webcam.Webcam;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

public class getWebcam {
    public static Webcam getPreferredWebcam() {
        List<Webcam> candidates = new ArrayList<>();
        Webcam preferred = Webcam.getWebcamByName("C270 HD WEBCAM /dev/video2");
        if (preferred != null) candidates.add(preferred);
        Webcam def = Webcam.getDefault();
        if (def != null) candidates.add(def);
        candidates.addAll(Webcam.getWebcams());

        for (Webcam cam : candidates) {
            try { cam.open(); return cam; } catch (Exception ignored) { }
        }
        System.err.println("No usable webcam found. Plug one in (on Linux, check you're in the 'video' group).");
        System.exit(1);
        return null;
    }
}
