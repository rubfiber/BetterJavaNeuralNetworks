package org.example.Usage.Recurrent;

import java.util.Random;
import java.util.Arrays;

public class SampleData {
    private SampleData() {}


    public static float[][][][] delayedEcho(int numSequences, int seqLength, int delay, long seed) {
        Random rnd = new Random(seed);
        float[][][] input = new float[numSequences][seqLength][1];
        float[][][] target = new float[numSequences][seqLength][1];
        for (int s = 0; s < numSequences; s++) {
            for (int t = 0; t < seqLength; t++) {
                input[s][t][0] = rnd.nextBoolean() ? 1f : 0f;
                target[s][t][0] = t >= delay ? input[s][t - delay][0] : 0f;
            }
        }
        return new float[][][][]{input, target};
    }


    public static float[][][][] runningParity(int numSequences, int seqLength, long seed) {
        Random rnd = new Random(seed);
        float[][][] input = new float[numSequences][seqLength][1];
        float[][][] target = new float[numSequences][seqLength][1];
        for (int s = 0; s < numSequences; s++) {
            int parity = 0;
            for (int t = 0; t < seqLength; t++) {
                int bit = rnd.nextBoolean() ? 1 : 0;
                parity ^= bit;
                input[s][t][0] = bit;
                target[s][t][0] = parity;
            }
        }
        return new float[][][][]{input, target};
    }


    public static float[][][][] nextSymbol(int numSequences, int seqLength, int numSymbols, long seed) {
        Random rnd = new Random(seed);
        float[][][] input = new float[numSequences][seqLength][numSymbols];
        float[][][] target = new float[numSequences][seqLength][numSymbols];
        for (int s = 0; s < numSequences; s++) {
            int start = rnd.nextInt(numSymbols);
            for (int t = 0; t < seqLength; t++) {
                input[s][t][(start + t) % numSymbols] = 1f;
                target[s][t][(start + t + 1) % numSymbols] = 1f;
            }
        }
        return new float[][][][]{input, target};
    }

    private static void print(String name, float[][][][] data) {
        float[][][] in = data[0], out = data[1];
        System.out.printf("%s: input[%d][%d][%d], target[%d][%d][%d]%n", name,
                in.length, in[0].length, in[0][0].length,
                out.length, out[0].length, out[0][0].length);
        StringBuilder a = new StringBuilder(), b = new StringBuilder();
        for (int t = 0; t < in[0].length; t++) {
            a.append(java.util.Arrays.toString(in[0][t])).append(' ');
            b.append(java.util.Arrays.toString(out[0][t])).append(' ');
        }
        System.out.println("  seq 0 input : " + a);
        System.out.println("  seq 0 target: " + b);
    }

    public static void main(String[] args) {
        print("delayedEcho(delay=1)", delayedEcho(200, 8, 1, 42));
        print("runningParity", runningParity(200, 8, 42));
        print("nextSymbol(k=3)", nextSymbol(200, 8, 3, 42));
    }
}