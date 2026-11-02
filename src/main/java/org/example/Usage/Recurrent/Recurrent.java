package org.example.Usage.Recurrent;

import org.network.Recurrent.LSTM;
import org.network.Recurrent.RecurrentNetwork;

import java.util.Arrays;

public class Recurrent {
    public static final RecurrentNetwork rnn = new RecurrentNetwork(new int[]{2, 16, 1});
    public static final LSTM lstm = new LSTM(new int[]{5, 10, 10, 5});
    float[][][] X = new float[100][5][2]; //input 0, output 0
    float[][][] Y = new float[100][5][1];
    void main() throws InterruptedException {
//        while(true) {
//            try {
//                System.out.println(Arrays.toString(lstm.predict(new float[]{0.1f, 0.3f, 0.5f, 0.7f, 0.9f})));
//                Thread.sleep(1000);
//              } catch (InterruptedException e) {
//            System.out.println("Thread interrupted");
//            Thread.currentThread().interrupt();
//            break;
//        }
//
//        }
//
        rnn.train(X, Y, 200, 0.05f, 5.0f);

        rnn.resetState();
        for (int t = 0; t < 5; t++) {
            float[] out = rnn.predict(X[0][t]);
            System.out.println("Step " + t + " prediction: " + out[0] + " (target: " + Y[0][t][0] + ")");
        }    }
}
