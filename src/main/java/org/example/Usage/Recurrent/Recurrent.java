package org.example.Usage.Recurrent;

import org.network.Recurrent.LSTM;
import org.network.Recurrent.RecurrentNetwork;

import java.util.Arrays;

public class Recurrent {
    public static final RecurrentNetwork rnn = new RecurrentNetwork(new int[]{1, 16, 1});
    public static final LSTM lstm = new LSTM(new int[]{5, 10, 10, 5});
    float[][][] X = new float[100][5][2]; //input 0, output 0
    float[][][] Y = new float[100][5][1];

    void main() {
        float[][][][] trainData = SampleData.delayedEcho(200, 8, 1, 42);
        rnn.train(trainData[0], trainData[1], 100, 0.05f, 5.0f);

        float[][][][] test = SampleData.delayedEcho(100, 8, 1, 999);

        RnnCheck.evaluate(seq -> {
            rnn.resetState();
            float[][] out = new float[seq.length][];
            for (int t = 0; t < seq.length; t++) out[t] = rnn.predict(seq[t]);
            return out;
        }, test[0], test[1], 1);
    }
}
