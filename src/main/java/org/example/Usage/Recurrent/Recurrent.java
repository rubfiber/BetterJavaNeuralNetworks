package org.example.Usage.Recurrent;

import org.network.Recurrent.RecurrentNetwork;

import java.util.Arrays;

public class Recurrent {
    public static final RecurrentNetwork rnn = new RecurrentNetwork(new int[]{5, 10, 5});
    void main() {
        System.out.println(Arrays.toString(rnn.predict(new float[]{0.1f, 0.3f, 0.5f, 0.7f, 0.9f}, RecurrentNetwork.RecurrentTypes.MANY_TO_MANY)));
    }
}
