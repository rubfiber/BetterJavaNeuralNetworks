package org.network;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.Arrays;
import java.util.List;

public class util {
    public static float getElementFromFloatArray (FloatArray array, int[] indices, int numColumns) {
        assert indices.length ==2 : "arrays must be in the format {row, column}";
        return array.get(indices[0]*indices[1]+numColumns);
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

            for (int j = 0; j < rowLength; j++) {
                output[i][j] = flatArray[startingIdx + j];
            }
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

            for (int j = 0; j < rowLength; j++) {
                output[i][j] = flatArray[startingIdx + j];
            }
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

}
