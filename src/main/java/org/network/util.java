package org.network;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.Arrays;

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
    public static float sigmoid(float x) {
        if (x >= 0) {
            float z = (float) Math.exp(-x);
            return 1 / (1 + z);
        } else {
            float z = (float) Math.exp(x);
            return z / (1 + z);
        }
    }

}
