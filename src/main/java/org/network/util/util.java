package org.network.util;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class util {
    public static float getElementFromFloatArray(FloatArray array, int[] indices, int numColumns) {
        assert indices.length == 2 : "arrays must be in the format {row, column}";
        return array.get(indices[0] * indices[1] + numColumns);
    }

    public static float[] flatten(float[][] matrix) {
        if (matrix == null) {
            return new float[0];
        }
        int totalElements = 0;
        for (float[] row : matrix) {
            if (row != null) {
                totalElements += row.length;
            }
        }
        float[] flatArray = new float[totalElements];
        int index = 0;
        for (float[] row : matrix) {
            if (row != null) {
                for (float value : row) {
                    flatArray[index++] = value;
                }
            }
        }

        return flatArray;
    }

    public static float[] flatten3D(float[][][] matrix) {
        if (matrix == null) {
            return new float[0];
        }
        int totalElements = 0;
        for (float[][]  row : matrix) {
            for (float[] col : row) {
                totalElements += col.length;
            }
        }
        float[] flatArray = new float[totalElements];
        int idx = 0;
        for (float[][]  row : matrix) {
            for (float[] col : row) {
                for (float value : col) {
                    flatArray[idx++] = value;
                }
            }
        }

        return flatArray;
    }
    public static float getIndexAt(float[] input, int x, int y, int z, int width, int height) {
        return x + (y * width) + (z * width * height);

    }

    public static float[][] deFlatten(float[] matrix, int[] key) {
        return deFlattenItems(matrix, key);
    }

    private static float[][] deFlattenItems(float[] matrix, int[] key) {
        int depth = key.length - 1;
        float[][] output = new float[depth][];
        int startingIdx = 0;

        for (int i = 0; i < depth; i++) {
            int rowLength = key[i] * key[i + 1];
            output[i] = new float[rowLength];

            System.arraycopy(matrix, startingIdx, output[i], 0, rowLength);
            startingIdx += rowLength;
        }
        return output;
    }

    public static float[][] deflattenWeights(float[] flatArray, int[] layers) {
        if (flatArray == null || layers == null) return new float[0][0];

        return deFlattenItems(flatArray, layers);
    }

    public static float[][] deflattenBias(float[] flatArray, int[] layers) {
        if (flatArray == null || layers == null) return new float[0][0];

        int depth = layers.length - 1;
        float[][] output = new float[depth][];
        int startingIdx = 0;

        for (int i = 0; i < depth; i++) {
            int rowLength = layers[i + 1];
            output[i] = new float[rowLength];

            System.arraycopy(flatArray, startingIdx, output[i], 0, rowLength);
            startingIdx += rowLength;
        }
        return output;
    }

    public static float sigmoid(float x) {
        if (x >= 0) {
            float z = (float) Math.exp(-x);
            return 1 / (1 + z);
        } else {
            float z = (float) Math.exp(x);
            return z / (1 + z);
        }
    }

    public static float[][] toPrimativeFloatArray(List<List<Float>> list) {
        if (list == null) return null;

        float[][] array = new float[list.size()][];
        for (int i = 0; i < list.size(); i++) {
            List<Float> innerList = list.get(i);
            array[i] = new float[innerList.size()];
            for (int j = 0; j < innerList.size(); j++) {
                Float value = innerList.get(j);
                array[i][j] = (value != null) ? value : 0.0f;
            }
        }
        return array;
    }

    public static float[][][] convertToFloat(int[][][] image) {
        int width = image.length;
        int height = image[0].length;
        int channels = 3;
        float[][][] floatImage = new float[width][height][channels];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {


                int redInt = image[x][y][0];
                int greenInt = image[x][y][1];
                int blueInt = image[x][y][2];

                floatImage[0][y][x] = redInt / 255.0f;
                floatImage[1][y][x] = greenInt / 255.0f;
                floatImage[2][y][x] = blueInt / 255.0f;
            }
        }
        return floatImage;
    }

    public static int[][] padImage(int[][] image, int[] padding) {
        int width = image.length;
        int height = image[0].length;

        int wPadding = padding[0];
        int hPadding = padding[1];

        int[][] paddedImage = new int[width + wPadding][height + hPadding];

        int leftPadding = Math.floorDiv(wPadding, 2);
        int upPadding = Math.floorDiv(hPadding, 2);

        for (int x = 0; x < width; x++) {
            System.arraycopy(image[x], 0, paddedImage[x + leftPadding], upPadding, height);
        }

        return paddedImage;
    }
    public static float[][][] padImage(float[][][] image, int[] padding) {
        int channels = image.length;
        int height = image[0].length;
        int width = image[0][0].length;

        int wPadding = padding[0];
        int hPadding = padding[1];

        float[][][] paddedImage = new float[channels][height + hPadding][width + wPadding];

        int leftPadding = Math.floorDiv(wPadding, 2);
        int upPadding = Math.floorDiv(hPadding, 2);

        for (int x = 0; x < width; x++) {
            System.arraycopy(image[x], 0, paddedImage[x + leftPadding], upPadding, height);
        }

        return paddedImage;
    }

    public static float[] softmax(float[] logits) {

        double max = logits[0];
        for (double val : logits) {
            if (val > max) {
                max = val;
            }
        }

        double[] expValues = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            expValues[i] = Math.exp(logits[i] - max);
            sum += expValues[i];
        }

        float[] probabilities = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = (float) (expValues[i] / sum);
        }

        return probabilities;
    }

    public static float[][] matrixMultiply(float[][] A, float[][] B, int m, int p, int n) {
        float[][] C = new float[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float dotProduct = 0.0f;
                for (int k = 0; k < p; k++) {
                    dotProduct += A[i][k] * B[k][j];
                }
                C[i][j] = dotProduct;
            }
        }

        return C;
    }
    public static int getIndexOfMax(float[] array) {
        if (array == null || array.length == 0) {
            return -1;
        }

        return IntStream.range(0, array.length)
                .reduce((i, j) -> array[i] > array[j] ? i : j)
               .getAsInt();
    }
    public static float relu(float output) {
        return Math.max(0, output);
    }

    public static float leakyRelu(float output) {
        return relu(output);
    }
    public static float linear(float output) {return output;}

    private static int minCh(int p) {
        return Math.min((p >> 16) & 0xFF, Math.min((p >> 8) & 0xFF, p & 0xFF));
    }

    private static void seedBg(BufferedImage im, boolean[] bg, java.util.ArrayDeque<Integer> q, int x, int y) {
        int w = im.getWidth(), h = im.getHeight();
        if (x < 0 || y < 0 || x >= w || y >= h || bg[y * w + x]) return;
        if (minCh(im.getRGB(x, y)) >= 235) { bg[y * w + x] = true; q.add(y * w + x); }
    }

    /** Returns an ARGB copy of src with the border-connected white made transparent. */
    public static BufferedImage keyOutWhite(BufferedImage src) {
        int w = src.getWidth(), h = src.getHeight();
        boolean[] bg = new boolean[w * h];
        java.util.ArrayDeque<Integer> q = new java.util.ArrayDeque<>();
        for (int x = 0; x < w; x++) { seedBg(src, bg, q, x, 0); seedBg(src, bg, q, x, h - 1); }
        for (int y = 0; y < h; y++) { seedBg(src, bg, q, 0, y); seedBg(src, bg, q, w - 1, y); }
        while (!q.isEmpty()) {
            int p = q.poll(), x = p % w, y = p / w;
            seedBg(src, bg, q, x - 1, y); seedBg(src, bg, q, x + 1, y);
            seedBg(src, bg, q, x, y - 1); seedBg(src, bg, q, x, y + 1);
        }

        final int erode = 1;                    // raise to 2 if a fringe remains
        for (int it = 0; it < erode; it++) {
            boolean[] next = bg.clone();
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int idx = y * w + x;
                    if (bg[idx]) continue;
                    if ((x > 0 && bg[idx - 1]) || (x < w - 1 && bg[idx + 1])
                            || (y > 0 && bg[idx - w]) || (y < h - 1 && bg[idx + w])) next[idx] = true;
                }
            bg = next;
        }

        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                if (bg[idx]) { out.setRGB(x, y, 0); continue; }
                boolean ring = (x > 0 && bg[idx - 1]) || (x < w - 1 && bg[idx + 1])
                        || (y > 0 && bg[idx - w]) || (y < h - 1 && bg[idx + w]);
                int alpha = ring ? 170 : 255;   // slightly soft outer edge, like a camera
                out.setRGB(x, y, (alpha << 24) | (src.getRGB(x, y) & 0xFFFFFF));
            }
        }
        return out;
    }
    public static float[] lowFreqTexture(int n, int cells, float amp, Random rand) {
        float[][] grid = new float[cells + 1][cells + 1];
        for (float[] row : grid) for (int j = 0; j < row.length; j++) row[j] = (rand.nextFloat() * 2f - 1f) * amp;
        float[] out = new float[n * n];
        for (int y = 0; y < n; y++) {
            float fy = (float) y / (n - 1) * cells;
            int iy = Math.min((int) fy, cells - 1);
            float ty = fy - iy;
            for (int x = 0; x < n; x++) {
                float fx = (float) x / (n - 1) * cells;
                int ix = Math.min((int) fx, cells - 1);
                float tx = fx - ix;
                float top = grid[iy][ix] * (1 - tx) + grid[iy][ix + 1] * tx;
                float bot = grid[iy + 1][ix] * (1 - tx) + grid[iy + 1][ix + 1] * tx;
                out[y * n + x] = top * (1 - ty) + bot * ty;
            }
        }
        return out;
    }
}
