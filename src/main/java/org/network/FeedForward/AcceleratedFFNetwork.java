package org.network.FeedForward;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import static org.network.util.sigmoid;

public class AcceleratedFFNetwork extends FeedForwardNetwork {
    public AcceleratedFFNetwork(int inputSize, int hiddenLayerWidth, int hiddenLayerDepth, int outputSize) {
        super(inputSize, hiddenLayerWidth, hiddenLayerDepth, outputSize);
    }
    public static void computeLayer(FloatArray input, FloatArray weights, FloatArray bias, FloatArray output, int inputSize, int outputSize, boolean isOutputLayer) {
        for (@Parallel int neuron = 0; neuron < inputSize; neuron++) {
            float sum = 0.0f;
            for (int in = 0; in < inputSize; in++) {
                sum += input.get(in) * weights.get(neuron * inputSize + in);
            }
            sum += bias.get(neuron);
            if (isOutputLayer) {
                output.set(neuron, (float) Math.tanh(sum));
            } else {
                output.set(neuron, sigmoid(sum));
            }
        }
    }

    public TaskGraph buildPredictionTaskGraph(FloatArray input, FeedForwardNetwork network) {
        hiddenLayerWeights = network.hiddenLayerWeights;
        float[][] hiddenLayerBias = network.hiddenLayerBias;
        float[][] hiddenLayerOutputs = network.hiddenLayerOutputs;
        float[] lastLayerWeights = network.outputLayerWeights;
        float[] lastLayerBias = network.outputLayerBias;
        float[] lastLayerOutputs = new float[outputSize];
        int inputSize = network.inputSize;
        // ...

        TaskGraph taskGraph = new TaskGraph("mlp_predict");

        FloatArray firstWeights = FloatArray.fromArray(hiddenLayerWeights[0]);
        FloatArray firstBias = FloatArray.fromArray(hiddenLayerBias[0]);
        FloatArray firstOutputs = FloatArray.fromArray(hiddenLayerOutputs[0]);

        taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, input, firstWeights, firstBias);
        taskGraph.task("Layer0",
                AcceleratedFFNetwork::computeLayer,
                input,
                firstWeights,
                firstBias,
                firstOutputs,
                inputSize,
                hiddenLayerWidth,
                false
        );
        for (int layer = 1; layer < hiddenLayerDepth-1; layer++) {
            String name = "Layer" + layer;
            FloatArray layerInputs = FloatArray.fromArray(hiddenLayerOutputs[layer-1]);
            FloatArray layerWeights = FloatArray.fromArray(hiddenLayerWeights[layer]);
            FloatArray layerBias = FloatArray.fromArray(hiddenLayerBias[layer]);
            FloatArray layerOutputs = FloatArray.fromArray(hiddenLayerOutputs[layer]);
            taskGraph.task(name,
                    AcceleratedFFNetwork::computeLayer,
                    layerInputs, //output of last layer is input of this layer
                    layerWeights,
                    layerBias,
                    layerOutputs,
                    hiddenLayerWidth,
                    hiddenLayerDepth,
                    false
            );


        }
        FloatArray finalLayerInputs =  FloatArray.fromArray(hiddenLayerOutputs[hiddenLayerOutputs.length-1]);
        FloatArray finalLayerWeights = FloatArray.fromArray(lastLayerWeights);
        FloatArray finalLayerBias = FloatArray.fromArray(lastLayerBias);
        FloatArray finalLayerOutputs = FloatArray.fromArray(lastLayerOutputs);
        taskGraph.task("FinalLayer",
                AcceleratedFFNetwork::computeLayer,
                finalLayerInputs,
                finalLayerWeights,
                finalLayerBias,
                finalLayerOutputs,
                hiddenLayerWidth,
                outputSize,
                false
        );

        taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, finalLayerOutputs);

        return taskGraph;
    }
    public static float[] acceleratedOutputLayerPredict( int hiddenLayerWidth, int lastLayerIdx, int outputSize, float[] outLayerWeights, float[] outLayerBias,  float[][] hiddenLayerOutputs) {
        float[] lastHiddenOutputs = hiddenLayerOutputs[lastLayerIdx - 1];
        int outWeightIndex = 0;
        float[] outLayerOutputs = hiddenLayerOutputs[lastLayerIdx];

        for (@Parallel int neuron = 0; neuron < outputSize; neuron++) {
            float sum = 0.0f;
            for (int input = 0; input < hiddenLayerWidth; input++) {
                sum += lastHiddenOutputs[input] * outLayerWeights[outWeightIndex++];
            }
            sum += outLayerBias[neuron];

            outLayerOutputs[neuron] = (float) Math.tanh(sum);
        }
        return  outLayerOutputs;
    }
}
