package org.example.Usage.Recurrent;

import java.util.Arrays;
import java.util.function.Function;


public class RnnCheck {
    private RnnCheck() {}

    public static double evaluate(Function<float[][], float[][]> predict,
                                  float[][][] inputs, float[][][] targets, int skipSteps) {
        int correct = 0, total = 0;
        double sqErr = 0;
        int shown = 0;

        for (int s = 0; s < inputs.length; s++) {
            float[][] out = predict.apply(inputs[s]);
            if (out.length != targets[s].length) {
                throw new IllegalStateException("predict returned " + out.length
                        + " steps but the target has " + targets[s].length);
            }
            StringBuilder inStr = new StringBuilder(), tgtStr = new StringBuilder(), outStr = new StringBuilder();
            for (int t = 0; t < out.length; t++) {
                boolean ok = isCorrect(out[t], targets[s][t]);
                for (int k = 0; k < out[t].length; k++) {
                    double d = out[t][k] - targets[s][t][k];
                    if (t >= skipSteps) sqErr += d * d;
                }
                if (t >= skipSteps) {
                    total++;
                    if (ok) correct++;
                }
                if (shown < 3) {
                    inStr.append(compact(inputs[s][t])).append(' ');
                    tgtStr.append(compact(targets[s][t])).append(' ');
                    outStr.append(String.format("%s%s ", compact(out[t]), ok ? "" : "!"));
                }
            }
            if (shown < 3) {
                System.out.println("  input : " + inStr);
                System.out.println("  target: " + tgtStr);
                System.out.println("  output: " + outStr + "   (! = wrong)");
                shown++;
            }
        }
        double acc = total == 0 ? 0 : (double) correct / total;
        int outputs = targets[0][0].length;
        System.out.printf("accuracy %.1f%% (%d/%d steps) | mean squared error %.4f%n",
                acc * 100, correct, total, sqErr / Math.max(1, (long) total * outputs));
        return acc;
    }

    private static boolean isCorrect(float[] out, float[] target) {
        if (target.length == 1) return (out[0] >= 0.5f) == (target[0] >= 0.5f);
        return argmax(out) == argmax(target);
    }

    private static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    private static String compact(float[] v) {
        if (v.length == 1) return String.format("%.2f", v[0]);
        return Arrays.toString(Arrays.stream(toDouble(v)).mapToObj(x -> String.format("%.1f", x)).toArray());
    }

    private static double[] toDouble(float[] v) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) d[i] = v[i];
        return d;
    }

    static void main() {
        float[][][][] data = SampleData.delayedEcho(100, 8, 1, 999);

        System.out.println("A predictor that echoes perfectly:");
        evaluate(seq -> {
            float[][] out = new float[seq.length][1];
            for (int t = 1; t < seq.length; t++) out[t][0] = seq[t - 1][0];
            return out;
        }, data[0], data[1], 1);

        System.out.println("\nA predictor that always says 0.5 (what an untrained network looks like):");
        evaluate(seq -> {
            float[][] out = new float[seq.length][1];
            for (float[] row : out) row[0] = 0.5f;
            return out;
        }, data[0], data[1], 1);
    }
}