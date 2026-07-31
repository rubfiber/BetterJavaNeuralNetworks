package org.example;

import org.network.FeedForward.FeedForwardNetwork;

import java.util.Arrays;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    static FeedForwardNetwork network = new FeedForwardNetwork(3, 4, 2, 4);
    public static void main(String[] args) {
        System.out.println(Arrays.toString(network.predict(new float[]{0.2f, 0.3f, 0.2f})));
    }
}