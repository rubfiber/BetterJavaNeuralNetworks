package org.example.Usage.Recurrent;

import org.network.Recurrent.RecurrentNetwork;

import java.util.Arrays;

public class Recurrent {
    public static final RecurrentNetwork rnn = new RecurrentNetwork(new int[]{5, 10, 10, 5});
    void main() throws InterruptedException {
        while(true) {
            System.out.println(Arrays.toString(rnn.predict(new float[]{0.1f, 0.3f, 0.5f, 0.7f, 0.9f})));
            Thread.sleep(1000);
        }
    }
}
