package org.network;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.List;

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
                // Safe unboxing; handles null checking if necessary
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
    public static float relu(float output) {
        return Math.max(0, output);
    }

    public static float leakyRelu(float output) {
        return relu(output);
    }
    public static float linear(float output) {return output;}
}
