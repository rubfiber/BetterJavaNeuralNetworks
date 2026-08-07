package org.network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class Dataset {

    List<List<Float>> inputData = new ArrayList<>();
    List<List<Float>> outputData = new ArrayList<>();

    public Dataset(float[][] initialInput, float[][] initialOutput) {
        inputData = Arrays.stream(initialInput)
                .map(innerArray -> IntStream.range(0, innerArray.length)
                        .mapToObj(i -> innerArray[i])
                        .toList()
                )
                .toList(); //make list from initialdata
        outputData = Arrays.stream(initialOutput)
                .map(innerArray -> IntStream.range(0, innerArray.length)
                        .mapToObj(i -> innerArray[i])
                        .toList()
                )
                .toList();
    }

    public Dataset() {
        inputData = new ArrayList<>();
        outputData = new ArrayList<>();
    }

    public float[][] getInputData() {
        return util.toPrimativeFloatArray(inputData);
    }

    public float[][] getOutputData() {
        return util.toPrimativeFloatArray(outputData);
    }

    public void setInputData(float[][] data) {
        this.inputData = Arrays.stream(data)
                .map(innerArray -> IntStream.range(0, innerArray.length)
                        .mapToObj(i -> innerArray[i])
                        .toList()
                )
                .toList();
    }

    public void setOutputData(float[][] data) {
        this.outputData = Arrays.stream(data)
                .map(innerArray -> IntStream.range(0, innerArray.length)
                        .mapToObj(i -> innerArray[i])
                        .toList()
                )
                .toList();
    }

    public void addInputData(float data) {
        inputData.add(List.of(data));
    }
    public void addOutput(float data) {
        outputData.add(List.of(data));
    }
}