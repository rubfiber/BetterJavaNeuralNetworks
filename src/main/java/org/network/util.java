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

    public static float[][] deflattenWeights(float[] flatArray, int[] layers) {
        if (flatArray == null || layers == null) return new float[0][0];

        int depth = layers.length - 1;
        float[][] output = new float[depth][];
        int startingIdx = 0;

        for (int i = 0; i < depth; i++) {
            int rowLength = layers[i] * layers[i + 1];
            output[i] = new float[rowLength];

            System.arraycopy(flatArray, startingIdx, output[i], 0, rowLength);
            startingIdx += rowLength;
        }
        return output;
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

    public static float[][][] padImage(float[][][] image, int[] padding) {
        int width = image.length;
        int height = image[0].length;
        int channels = image[0][0].length; //should be 3 cuz RGB

        int wPadding = padding[0];
        int hPadding = padding[1];

        float[][][] paddedImage = new float[width + wPadding][height + hPadding][channels];

        int leftPadding = Math.floorDiv(wPadding, 2);
        int upPadding = Math.floorDiv(hPadding, 2);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                paddedImage[x + leftPadding][y + upPadding] = image[x][y];

            }
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

    public static float relu(float output) {
        return Math.max(0, output);
    }

    public static float leakyRelu(float output) {
        return relu(output);
    }
    public static float linear(float output) {return output;}
}
