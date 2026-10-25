package org.network.Convolutional;

import org.network.FeedForward.Activation;
import org.network.FeedForward.FeedForwardNetwork;
import org.network.util.util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.network.util.util.*;

public class ConvolutionalNetwork {

    private static final Logger log = LoggerFactory.getLogger(ConvolutionalNetwork.class);
    //filter size is 3x3 to avoid 6 nested loops
    public int filterSize;
    public int[] padding;
    public int batchSize;

    public ConvolutionalLayer[] cnnLayers;
    public FeedForwardNetwork network;
    public int[] filters;
    public int gridSize = 4;


    public ConvolutionalNetwork(filterBuilder builder, int[] FFNLayers, int[] filters, int[] inputChannels, int[] stride) {
        this.filterSize = FFNLayers.length;
        this.filters = filters;
        this.batchSize = builder.batchSize;
        this.padding = builder.padding;

        network = new FeedForwardNetwork(FFNLayers);
        network.fillHiddenActivations(Activation.RELU);
        network.fillOutputActivation(Activation.SOFTMAX);

        cnnLayers = new ConvolutionalLayer[filters.length];
        for (int i = 0; i < filters.length; i++) {
            cnnLayers[i] = new ConvolutionalLayer(inputChannels[i], filters[i], stride[i], this.padding, gridSize, 3);
        }
    }


    public float[][][] forwardPass(float[][][] initialImage) {
        float[][][] currentMap = initialImage; //raw image

        for (int i = 0; i < cnnLayers.length; i++) {
            currentMap = cnnLayers[i].im2col(currentMap, cnnLayers[i].weights, cnnLayers[i].stride, cnnLayers[i].padding[0]);

            // pooling happens every 2nd layer
            if ((i + 1) % 2 == 0) {
                currentMap = cnnLayers[i].maxPool(currentMap, 2, 2);
            }
        }
        return currentMap;
    }

    public float[] predict(float[][][] image) {
        float[][][] cnnOutput = forwardPass(image);
        float[] flattenedImage = util.flatten3D(cnnOutput);
        return network.predict(flattenedImage);
    }

    int lastCnnHeight;
    int lastCnnWidth;
    int lastCnnFilters;
    private static final float COORD_WEIGHT = 2f, NOOBJ_WEIGHT = 0.2f;

    private float[] yoloWeights(float[] target) {
        int cells = gridSize * gridSize;
        int fpc = target.length / cells;
        float[] w = new float[target.length];
        for (int c = 0; c < cells; c++) {
            int s = c * fpc;
            if (target[s] > 0.5f) {
                w[s] = 2f;
                for (int k = 1; k <= 4; k++) w[s + k] = COORD_WEIGHT;
                for (int k = 5; k < fpc; k++) {
                    w[s + k] = (target[s + k] > 0.5f) ? 8.0f : 0.5f;
                }
            } else {
                w[s] = NOOBJ_WEIGHT;
            }
        }
        return w;
    }
    public void train(float[][][] image, float[] desiredOutput) {
        float[][][] cnnOutput = forwardPass(image);
        int lastNumFilters = cnnOutput.length;         // depth/channels
        int lastHeight = cnnOutput[0].length;      // height
        int lastWidth = cnnOutput[0][0].length;   // width

        this.lastCnnHeight = lastHeight;
        this.lastCnnWidth = lastWidth;
        this.lastCnnFilters = lastNumFilters;


        float[] flatData = util.flatten3D(cnnOutput);
        float[] targetForFfn = desiredOutput;
        if (useYoloLoss) {
            float[] w = yoloWeights(desiredOutput);
            float[] out = network.predict(flatData);
            targetForFfn = new float[desiredOutput.length];
            for (int i = 0; i < out.length; i++) {
                targetForFfn[i] = out[i] - w[i] * (out[i] - desiredOutput[i]);
            }
        }
        float[] flatError1D = network.accumulateGradientsForCnn(flatData, targetForFfn);
        float[][][] current3DError = unflattenError(flatError1D, lastNumFilters, lastHeight, lastWidth);
        for (int layer = cnnLayers.length - 1; layer >= 0; layer--) {
            if ((layer + 1) % 2 == 0) { //backprop pooling happens in the same place as forward pass
                current3DError = cnnLayers[layer].poolingBackprop(current3DError);
            }
            current3DError = cnnLayers[layer].accumulateGradients(current3DError);
        }
    }

    public void trainDataset(float[][][][] imageInputs, float[][] desiredOutputs, float learningRate, int batchSize, int epochs) {
        int numSamples = imageInputs.length;

        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }
        Random random = new Random();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {
                int swap = random.nextInt(numSamples);
                int temp = indices[i];
                indices[i] = indices[swap];
                indices[swap] = temp;
            }

            for (int sample = 0; sample < numSamples; sample += batchSize) {
                int currentBatchSize = Math.min(batchSize, numSamples - sample);

                for (int b = 0; b < currentBatchSize; b++) {
                    int sampleIdx = indices[sample + b]; // Grab shuffled image index

                    float[][][] image = imageInputs[sampleIdx];
                    float[] target = desiredOutputs[sampleIdx];

                    this.train(image, target);
                }
                network.applyBatch(learningRate, currentBatchSize);

                for (ConvolutionalLayer layer : cnnLayers) {
                    layer.applyBatch(learningRate, currentBatchSize);
                }
            }
            System.out.println("Epoch: " + epoch);
            if (useYoloLoss) printDiagnostics(imageInputs, desiredOutputs, 100);
        }

    }

    public boolean useYoloLoss = true;

    public void printDiagnostics(float[][][][] images, float[][] targets, int maxSamples) {
        int cells = gridSize * gridSize;
        int fpc = targets[0].length / cells;      // features per cell
        int nCls = fpc - 5;                       // class slots per cell
        int n = Math.min(maxSamples, images.length);

        double loss = 0, confAtObj = 0, confEmpty = 0;
        double trueClassProb = 0, topClassProb = 0;
        int objSamples = 0, cellHits = 0, nearHits = 0, classHits = 0;

        for (int s = 0; s < n; s++) {
            int idx = (int) ((long) s * images.length / n);   // spread across the dataset
            float[] pred = predict(images[idx]);
            float[] t = targets[idx];

            // overall loss (unweighted squared error)
            for (int k = 0; k < pred.length; k++) {
                double d = pred[k] - t[k];
                loss += d * d;
            }

            // find the true cell and the most confident predicted cell
            int trueCell = -1, bestCell = 0;
            float bestConf = -1f;
            for (int c = 0; c < cells; c++) {
                int base = c * fpc;
                if (pred[base] > bestConf) {
                    bestConf = pred[base];
                    bestCell = c;
                }
                if (t[base] > 0.5f) {
                    trueCell = c;
                    confAtObj += pred[base];
                } else {
                    confEmpty += pred[base];
                }
            }
            if (trueCell < 0) continue;   // unlabeled sample (no object target)

            objSamples++;

            // localization
            if (bestCell == trueCell) cellHits++;
            int dx = Math.abs(bestCell % gridSize - trueCell % gridSize);
            int dy = Math.abs(bestCell / gridSize - trueCell / gridSize);
            if (Math.max(dx, dy) <= 1) nearHits++;

            // classification at the true cell (independent of localization)
            int cb = trueCell * fpc + 5;
            int predClass = 0, trueClass = 0;
            for (int k = 1; k < nCls; k++) {
                if (pred[cb + k] > pred[cb + predClass]) predClass = k;
                if (t[cb + k] > t[cb + trueClass]) trueClass = k;
            }
            if (predClass == trueClass) classHits++;
            trueClassProb += pred[cb + trueClass];
            topClassProb += pred[cb + predClass];
        }

        int emptyCells = Math.max(1, n * cells - objSamples);
        int o = Math.max(1, objSamples);
        System.out.printf(
                "   loss %.4f | conf@object %.3f | conf@empty %.3f | cell %d/%d (within 1: %d) | "
                        + "class %d/%d (chance %.0f%%) | p(true class) %.3f | p(top class) %.3f%n",
                loss / n,
                confAtObj / o,
                confEmpty / emptyCells,
                cellHits, objSamples, nearHits,
                classHits, objSamples, 100.0 / nCls,
                trueClassProb / o,
                topClassProb / o);
    }

    private float[][][] unflattenError(float[] flatError1D, int numFilters, int height, int width) {
        float[][][] FFNError3D = new float[cnnLayers[cnnLayers.length - 1].numFilters][height][width];
        int idx = 0;
        for (int filter = 0; filter < numFilters; filter++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    FFNError3D[filter][y][x] = flatError1D[idx];
                    idx++;
                }
            }
        }
        return FFNError3D;
    }

    public static class filterBuilder {

        public int[][] inputSize;
        public int[] padding;
        public int batchSize;
        public int kernelSize;

        public filterBuilder(int[][] inputSize, int[] padding, int batchSize, int kernelSize) {
            this.inputSize = inputSize;
            this.padding = padding;
            this.batchSize = batchSize;
            this.kernelSize = kernelSize;
        }
    }

    public static class ConvolutionalLayer {
        int poolLastInHeight;
        int poolLastInWidth;

        float[][][][] weights;


        int filterSize = 3;
        public int[] padding;
        float[] biases;
        int stride;

        int kernelSize;

        float[][] lastInputColumn;
        float[][][] lastOutput;
        int numFilters;
        int gridSize;

        public ConvolutionalLayer(int inChannels, int numFilters, int stride, int[] padding, int gridSize, int kernelSize) {
            this.numFilters = numFilters;
            this.stride = stride;
            this.padding = padding;
            this.weights = new float[numFilters][inChannels][3][3];
            this.biases = new float[numFilters];
            this.kernelSize = kernelSize;
            this.gridSize = gridSize;


            float stdDev = (float) Math.sqrt(2.0 / (inChannels * filterSize * filterSize));
            Random rand = new Random();

            for (int filter = 0; filter < numFilters; filter++) {
                this.biases[filter] = 0.0f;
                for (int c = 0; c < inChannels; c++) {
                    for (int y = 0; y < filterSize; y++) {
                        for (int x = 0; x < filterSize; x++) {
                            this.weights[filter][c][y][x] = (float) rand.nextGaussian() * stdDev;
                        }
                    }
                }
            }
        }

        float[][] x_col;

        int lastInHeight;
        int lastInWidth;
        float[][] lastWeightRow;

        public float[][][] im2col(float[][][] input, float[][][][] weights, int stride, int pad) { //channels, col, row
            int inChannels = input.length;
            int inHeight = input[0].length;
            int inWidth = input[0][0].length;

            int k_height = weights[0][0].length;
            int k_width = weights[0][0][0].length;
            int outHeight = (inHeight - kernelSize + padding[0] * 2) / stride + 1;
            int outWidth = (inWidth - kernelSize + padding[1] * 2) / stride + 1;
            float[][][] paddedInput = applyPadding(input, inChannels, inHeight, inWidth, pad);

            int filters = weights.length;
            int patchRows = inChannels * k_height * k_width;
            int patchColumns = outHeight * outWidth;
            float[][] inputColumn = new float[patchRows][patchColumns];
            lastInputColumn = inputColumn;
            x_col = inputColumn;
            int colIdx = 0;
            for (int y = 0; y < outHeight; y++) {
                for (int x = 0; x < outWidth; x++) {
                    int startX = x * stride;
                    int startY = y * stride;

                    int rowIdx = 0;
                    for (int channel = 0; channel < inChannels; channel++) {
                        for (int k_y = 0; k_y < k_height; k_y++) {
                            for (int k_x = 0; k_x < k_width; k_x++) {
                                inputColumn[rowIdx][colIdx] = paddedInput[channel][startY + k_y][startX + k_x];
                                rowIdx++;
                            }
                        }
                    }
                    colIdx++;
                }

            }
            float[][] weightRow = new float[filters][patchRows];

            for (int f = 0; f < filters; f++) {
                int idx = 0;
                for (int c = 0; c < inChannels; c++) {
                    for (int ky = 0; ky < k_height; ky++) {
                        for (int kx = 0; kx < k_width; kx++) {
                            weightRow[f][idx++] = weights[f][c][ky][kx];
                        }
                    }
                }
            }

            float[][] Y_column = matrixMultiply(weightRow, inputColumn, filters, patchRows, patchColumns);

            float[][][] output = new float[filters][outHeight][outWidth];

            for (int filter = 0; filter < filters; filter++) {
                float bias = biases[filter];
                int idxCol = 0;

                for (int y = 0; y < outHeight; y++) {
                    for (int x = 0; x < outWidth; x++) {

                        float sum = Y_column[filter][idxCol++];

                        float finalPixel = sum + bias;

                        output[filter][y][x] = (finalPixel > 0 ? finalPixel : finalPixel * 0.01f);
                    }
                }
            }
            lastOutput = output;
            lastInHeight = input[0].length;
            lastInWidth = input[0][0].length;
            lastWeightRow = weightRow;
            return output;
        }

        public float[][][] col2im(float[][] input, int inChannels, int inHeight, int inWidth) {
            int hPadded = inHeight + 2 * padding[0];
            int wPadded = inWidth + 2 * padding[1];
            float[][][] paddedImage = new float[inChannels][hPadded][wPadded];

            int stepsY = (inHeight - kernelSize + padding[0] * 2) / stride + 1;
            int stepsX = (inWidth - kernelSize + padding[1] * 2) / stride + 1;

            int colIdx = 0;

            for (int y = 0; y < stepsY; y++) {
                for (int x = 0; x < stepsX; x++) {

                    int startY = y * stride;
                    int startX = x * stride;

                    int rowIdx = 0;

                    for (int c = 0; c < inChannels; c++) {
                        for (int ky = 0; ky < kernelSize; ky++) {
                            for (int kx = 0; kx < kernelSize; kx++) {

                                paddedImage[c][startY + ky][startX + kx] += input[rowIdx][colIdx];


                                rowIdx++;
                            }
                        }
                    }
                    colIdx++;
                }
            }
            float[][][] image = new float[inChannels][inHeight][inWidth];
            for (int c = 0; c < inChannels; c++) {
                for (int h = 0; h < inHeight; h++) {
                    System.arraycopy(paddedImage[c][h + padding[0]], padding[0], image[c][h], 0, inWidth);
                }
            }
            return image;

        }

        int lastInChannels;
        float[][][] output;
        int[][][] maxIndicesX;
        int[][][] maxIndicesY;

        public float[][][] maxPool(float[][][] input, int poolSize, int stride) {
            int channels = input.length;
            int inWidth = input[0][0].length;
            int inHeight = input[0].length;
            int outHeight = inHeight / poolSize;
            int outWidth = inWidth / poolSize;

            this.lastInChannels = channels;


            this.poolLastInHeight = inHeight;
            this.poolLastInWidth = inWidth;

            output = new float[channels][outHeight][outWidth];
            maxIndicesX = new int[channels][outHeight][outWidth];
            maxIndicesY = new int[channels][outHeight][outWidth];


            for (int channel = 0; channel < channels; channel++) {
                for (int y = 0; y < outHeight; y++) {
                    for (int x = 0; x < outWidth; x++) {

                        float maxVal = Float.NEGATIVE_INFINITY;
                        int maxX = -1;
                        int maxY = -1;
                        for (int poolY = 0; poolY < poolSize; poolY++) {
                            for (int poolX = 0; poolX < poolSize; poolX++) {
                                int inputY = y * stride + poolY;
                                int inputX = x * stride + poolX;
                                float val = input[channel][inputY][inputX];

                                if (val > maxVal) {
                                    maxVal = val;
                                    maxX = inputX;
                                    maxY = inputY;
                                }
                            }
                        }
                        output[channel][y][x] = maxVal;
                        maxIndicesY[channel][y][x] = maxY;
                        maxIndicesX[channel][y][x] = maxX;
                    }
                }
            }
            return output;
        }

        float[][] weightGrads;
        float[] biasGrads;

        public float[][][] accumulateGradients(float[][][] incomingError) {
            int filters = incomingError.length;
            int outHeight = incomingError[0].length;
            int outWidth = incomingError[0][0].length;
            int patchColumns = outHeight * outWidth;
            int inChannels = weights[0].length;
            int patchRows = inChannels * filterSize * filterSize;


            float[][] dy_col = new float[filters][patchColumns];

            for (int filter = 0; filter <= filters - 1; filter++) {
                int colIdx = 0;
                for (int y = 0; y < outHeight; y++) {
                    for (int x = 0; x < outWidth; x++) {
                        dy_col[filter][colIdx++] = incomingError[filter][y][x];
                    }
                }
            }
            float[][] dz_col = new float[filters][patchColumns];
            for (int filter = 0; filter < filters; filter++) {
                for (int col = 0; col < patchColumns; col++) {
                    int y = col / outWidth;
                    int x = col % outWidth;

                    float output = this.lastOutput[filter][y][x];
                    float derivative = (output > 0) ? 1f : 0.01f;

                    dz_col[filter][col] = dy_col[filter][col] * derivative;

                }
            }
            //bias calculation
            // NOTE: lazily allocate (don't reset) so gradients ACCUMULATE across every image
            // in a batch; applyBatch() is what zeroes these out again after applying the update.
            if (biasGrads == null) biasGrads = new float[filters];
            for (int filter = 0; filter <= filters - 1; filter++) {
                float biasErrorSum = 0;

                for (int col = 0; col <= patchColumns - 1; col++) {
                    biasErrorSum += dz_col[filter][col];
                }
                biasGrads[filter] += biasErrorSum;
                // biases[filter] -= (learningRate * biasGrads[filter]);
            }
            //weights calculation
            if (weightGrads == null) weightGrads = new float[filters][patchRows];
            for (int filter = 0; filter <= filters - 1; filter++) {
                for (int row = 0; row <= patchRows - 1; row++) {
                    float weightErrorSum = 0;
                    for (int col = 0; col <= patchColumns - 1; col++) {
                        float error = dz_col[filter][col];
                        weightErrorSum += error * lastInputColumn[row][col];
                    }
                    weightGrads[filter][row] += weightErrorSum;
                }
            }
            //input error
            float[][] dx_col = new float[patchRows][patchColumns];
            for (int row = 0; row <= patchRows - 1; row++) {
                for (int col = 0; col <= patchColumns - 1; col++) {
                    float sum = 0;
                    for (int filter = 0; filter <= filters - 1; filter++) {
                        float weight = lastWeightRow[filter][row];
                        float error = dz_col[filter][col];

                        sum += error * weight;
                    }
                    dx_col[row][col] = sum;
                }
            }
            return col2im(dx_col, inChannels, this.lastInHeight, this.lastInWidth);

        }

        public float[][][] poolingBackprop(float[][][] incomingError) { //pooling layer has no weights and bias, only reduces
            int channels = incomingError.length;
            int inHeight = poolLastInHeight;
            int inWidth = poolLastInWidth;
            int outHeight = incomingError[0].length;
            int outWidth = incomingError[0][0].length;


            float[][][] inputGradients = new float[channels][inHeight][inWidth];

            for (int channel = 0; channel < channels; channel++) {
                for (int y = 0; y < outHeight; y++) {
                    for (int x = 0; x < outWidth; x++) {
                        int targetX = maxIndicesX[channel][y][x];
                        int targetY = maxIndicesY[channel][y][x];

                        inputGradients[channel][targetY][targetX] += incomingError[channel][y][x];
                    }
                }
            }
            return inputGradients;
        }

        public void applyBatch(float learningRate, int batchSize) {
            for (int filter = 0; filter < biases.length; filter++) {
                biases[filter] -= (learningRate / batchSize) * biasGrads[filter];
                biasGrads[filter] = 0.0f;
            }

            for (int filter = 0; filter < weights.length; filter++) {
                int idx = 0;
                for (int c = 0; c < weights[0].length; c++) {
                    for (int ky = 0; ky < weights[0][0].length; ky++) {
                        for (int kx = 0; kx < weights[0][0][0].length; kx++) {
                            // update weights
                            weights[filter][c][ky][kx] -= (learningRate / batchSize) * weightGrads[filter][idx];

                            weightGrads[filter][idx] = 0.0f;
                            idx++;
                        }
                    }
                }
            }
        }

        private static float[][][] applyPadding(float[][][] X, int C, int H, int W, int P) {
            if (P == 0) return X;
            float[][][] padded = new float[C][H + 2 * P][W + 2 * P];
            for (int c = 0; c < C; c++) {
                for (int y = 0; y < H; y++) {
                    System.arraycopy(X[c][y], 0, padded[c][y + P], P, W);
                }
            }
            return padded;
        }

        public float[][][][] getWeights() {
            return weights;
        }

        public void setWeights(float[][][] weights) {
            this.weights = new float[][][][]{weights};
        }

        public float[] getBiases() {
            return biases;
        }

        public void setBiases(float[] biases) {
            this.biases = biases;
        }

    }

    public float[][][][] floatImages;
    public float[][] outputs;
    public float[][] imagePos;

    public void addData(String dirPath, String[] possibleAnswers, int targetSize, float brightnessRange, float contrastRange, float saturationRange, float shiftRange, Color imgBackgroundColor) {

        if (shiftRange != 0) {
            network.setOutputActivationWithIndex(network.outputSize - 1, Activation.SIGMOID);
            network.setOutputActivationWithIndex(network.outputSize - 2, Activation.SIGMOID); // for X and Y since they're different from the classification prediction
        }
        Path[] imagePaths;

        // find all image files
        try (var stream = Files.walk(Paths.get(dirPath))) {
            imagePaths = stream
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().matches(".*\\.(png|jpg|jpeg|gif|bmp)$"))
                    .toArray(Path[]::new);
        } catch (IOException e) {
            log.error("Failed to scan directory: ", e);
            return;
        }

        if (imagePaths.length == 0) {
            System.out.println("No images found in " + dirPath);
            return;
        }

        System.out.println("queueing " + imagePaths.length + " images");

        floatImages = new float[imagePaths.length][3][targetSize][targetSize];
        outputs = new float[imagePaths.length][possibleAnswers.length];
        imagePos = new float[imagePaths.length][2]; // (x, y)

        Random rand = new Random();
        int maxShiftPixels = Math.round(targetSize * shiftRange);

        for (int i = 0; i < imagePaths.length; i++) {
            Path path = imagePaths[i];
            String fileName = path.getFileName().toString().toLowerCase();

            boolean matchFound = false;
            int bestMatchLength = -1;
            for (String possibleAnswer : possibleAnswers) {
                String candidate = possibleAnswer.toLowerCase();
                if (fileName.contains(candidate) && candidate.length() > bestMatchLength) {
                    bestMatchLength = candidate.length();
                }
            }
            if (!matchFound) {
                System.err.println(fileName + "' does not match any answers. Defaulting to 0");
            }

            try {
                BufferedImage rawImg = ImageIO.read(path.toFile());
                if (rawImg == null) continue;

                BufferedImage canvas = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
                Graphics2D g2d = canvas.createGraphics();

                // background color
                g2d.setColor(imgBackgroundColor);
                g2d.fillRect(0, 0, targetSize, targetSize);

                int shiftX = (maxShiftPixels > 0) ? rand.nextInt(maxShiftPixels * 2 + 1) - maxShiftPixels : 0;
                int shiftY = (maxShiftPixels > 0) ? rand.nextInt(maxShiftPixels * 2 + 1) - maxShiftPixels : 0;
                imagePos[i][0] = 0.5f + ((float) shiftX / targetSize);
                imagePos[i][1] = 0.5f + ((float) shiftY / targetSize);
                g2d.drawImage(rawImg, shiftX, shiftY, targetSize, targetSize, null);
                g2d.dispose();

                // change color of image slightly
                float bFactor = (rand.nextFloat() * 2f - 1f) * brightnessRange;
                float cFactor = 1.0f + (rand.nextFloat() * 2f - 1f) * contrastRange;
                float sFactor = 1.0f + (rand.nextFloat() * 2f - 1f) * saturationRange;

                // rgb bitwise
                for (int y = 0; y < targetSize; y++) {
                    for (int x = 0; x < targetSize; x++) {
                        int pixel = canvas.getRGB(x, y);

                        float red = ((pixel >> 16) & 0xFF) / 255.0f;
                        float green = ((pixel >> 8) & 0xFF) / 255.0f;
                        float blue = (pixel & 0xFF) / 255.0f;

                        // brightness for variation
                        red += bFactor;
                        green += bFactor;
                        blue += bFactor;

                        // contrast
                        red = (red - 0.5f) * cFactor + 0.5f;
                        green = (green - 0.5f) * cFactor + 0.5f;
                        blue = (blue - 0.5f) * cFactor + 0.5f;

                        // saturation
                        float gray = 0.299f * red + 0.587f * green + 0.114f * blue;
                        red = gray + (red - gray) * sFactor;
                        green = gray + (green - gray) * sFactor;
                        blue = gray + (blue - gray) * sFactor;

                        floatImages[i][0][y][x] = Math.clamp(red, 0.0f, 1.0f);
                        floatImages[i][1][y][x] = Math.clamp(green, 0.0f, 1.0f);
                        floatImages[i][2][y][x] = Math.clamp(blue, 0.0f, 1.0f);
                    }
                }

            } catch (IOException e) {
                log.error("Failed to process image: {}", path, e);
            }
        }
    }

    public List<BufferedImage> bgFrames = new ArrayList<>();
    private static final float X_SQUASH = 0.75f;



    private static BufferedImage cropToContent(BufferedImage im) {
        int w = im.getWidth(), h = im.getHeight(), x0 = w, y0 = h, x1 = -1, y1 = -1;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if ((im.getRGB(x, y) >>> 24) > 40) {
                    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
                }
        return x1 < 0 ? null : im.getSubimage(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    }

    private BufferedImage makeBackground(int size, Random rand) {
        BufferedImage bg = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        if (!bgFrames.isEmpty() && rand.nextFloat() < 0.8f) {
            BufferedImage src = bgFrames.get(rand.nextInt(bgFrames.size()));
            float s = 0.5f + rand.nextFloat() * 0.5f;
            int cw = Math.round(src.getWidth() * s), ch = Math.round(src.getHeight() * s);
            int x = rand.nextInt(src.getWidth() - cw + 1), y = rand.nextInt(src.getHeight() - ch + 1);
            boolean flip = rand.nextBoolean();
            Graphics2D g = bg.createGraphics();
            g.drawImage(src, flip ? size : 0, 0, flip ? 0 : size, size, x, y, x + cw, y + ch, null);
            g.dispose();
        } else {
            int base = 70 + rand.nextInt(160), grain = rand.nextInt(6);
            float[] blot = lowFreqTexture(size, 3 + rand.nextInt(4), 8 + rand.nextInt(16), rand);
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++) {
                    int lum = Math.clamp(Math.round(base + blot[y * size + x] + rand.nextInt(grain * 2 + 1) - grain), 0, 255);
                    bg.setRGB(x, y, (lum << 16) | (lum << 8) | lum);
                }
        }
        return bg;
    }

    public void addDataGrids(String dirPath, String[] possibleAnswers, int targetSize, float brightnessRange, float contrastRange, float shiftRange, boolean keyOutWhite) {

        if (shiftRange != 0) {
            network.setOutputActivationWithIndex(network.outputSize - 1, Activation.SIGMOID);
            network.setOutputActivationWithIndex(network.outputSize - 2, Activation.SIGMOID); // for X and Y since they're different from the classification prediction
        }
        Path[] imagePaths;

        // find all image files
        try (var stream = Files.walk(Paths.get(dirPath))) {
            imagePaths = stream
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().matches(".*\\.(png|jpg|jpeg|gif|bmp)$"))
                    .toArray(Path[]::new);
        } catch (IOException e) {
            log.error("Failed to scan directory: ", e);
            return;
        }

        if (imagePaths.length == 0) {
            System.out.println("No images found in " + dirPath);
            return;
        }

        System.out.println("queueing " + imagePaths.length + " images");

        floatImages = new float[imagePaths.length][3][targetSize][targetSize];

        int featuresPerCell = 1 + 4 + possibleAnswers.length;

        outputs = new float[imagePaths.length][gridSize * gridSize * featuresPerCell];

        Random rand = new Random();

        for (int i = 0; i < imagePaths.length; i++) {
            Path path = imagePaths[i];
            String fileName = path.getFileName().toString().toLowerCase();
            int classIndex = getClassIndex(possibleAnswers, path);
            if (classIndex == -1) {
                System.err.println(fileName + "' does not match any answers. Defaulting to 0");
            }

            try {
                BufferedImage rawImg = ImageIO.read(path.toFile());
                if (rawImg == null) continue;

                BufferedImage canvas = makeBackground(targetSize, rand);
                BufferedImage fruit = cropToContent((keyOutWhite) ? keyOutWhite(rawImg) : rawImg);
                if (fruit == null) continue;

                float scale = 0.25f + rand.nextFloat() * 0.65f;
                float effW = fruit.getWidth() * X_SQUASH, effH = fruit.getHeight();
                float k = scale * targetSize / Math.max(effW, effH);
                int dw = Math.max(4, Math.round(effW * k)), dh = Math.max(4, Math.round(effH * k));
                int dx = rand.nextInt(targetSize - dw + 1), dy = rand.nextInt(targetSize - dh + 1);
                boolean flip = rand.nextBoolean();

                Graphics2D g2d = canvas.createGraphics();
                g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                if (flip) g2d.drawImage(fruit, dx + dw, dy, -dw, dh, null);
                else g2d.drawImage(fruit, dx, dy, dw, dh, null);
                g2d.dispose();

                float cx = (dx + dw / 2f) / targetSize, cy = (dy + dh / 2f) / targetSize;
                int cellX = Math.min(gridSize - 1, (int) (cx * gridSize));
                int cellY = Math.min(gridSize - 1, (int) (cy * gridSize));
                int startIdx = (cellY * gridSize + cellX) * featuresPerCell;
                if (classIndex != -1) {
                    outputs[i][startIdx] = 1f;
                    outputs[i][startIdx + 1] = cx * gridSize - cellX;
                    outputs[i][startIdx + 2] = cy * gridSize - cellY;
                    outputs[i][startIdx + 3] = dw / (float) targetSize;
                    outputs[i][startIdx + 4] = dh / (float) targetSize;
                    outputs[i][startIdx + 5 + classIndex] = 1f;
                }
                // change color of image slightly
                float bFactor = (rand.nextFloat() * 2f - 1f) * brightnessRange;
                float cFactor = 1.0f + (rand.nextFloat() * 2f - 1f) * contrastRange;
                float sFactor = 0.25f + rand.nextFloat() * 1.15f;

                float rGain = 1f + (rand.nextFloat() * 2f - 1f) * 0.15f;
                float gGain = 1f + (rand.nextFloat() * 2f - 1f) * 0.10f;
                float bGain = 1f + (rand.nextFloat() * 2f - 1f) * 0.20f;
                float noise = rand.nextFloat() * 0.04f;
                // rgb bitwise
                for (int y = 0; y < targetSize; y++) {
                    for (int x = 0; x < targetSize; x++) {
                        int pixel = canvas.getRGB(x, y);

                        float red = ((pixel >> 16) & 0xFF) / 255.0f;
                        float green = ((pixel >> 8) & 0xFF) / 255.0f;
                        float blue = (pixel & 0xFF) / 255.0f;

                        red *= rGain;
                        green *= gGain;
                        blue *= bGain;

                        // brightness variation
                        red += bFactor;
                        green += bFactor;
                        blue += bFactor;

                        // contrast
                        red = (red - 0.5f) * cFactor + 0.5f;
                        green = (green - 0.5f) * cFactor + 0.5f;
                        blue = (blue - 0.5f) * cFactor + 0.5f;

                        // saturation
                        float gray = 0.299f * red + 0.587f * green + 0.114f * blue;
                        red = gray + (red - gray) * sFactor;
                        green = gray + (green - gray) * sFactor;
                        blue = gray + (blue - gray) * sFactor;


                        red += (float) rand.nextGaussian() * noise;
                        green += (float) rand.nextGaussian() * noise;
                        blue += (float) rand.nextGaussian() * noise;
                        floatImages[i][0][y][x] = Math.clamp(red, 0.0f, 1.0f);
                        floatImages[i][1][y][x] = Math.clamp(green, 0.0f, 1.0f);
                        floatImages[i][2][y][x] = Math.clamp(blue, 0.0f, 1.0f);
                    }
                }

            } catch (IOException e) {
                log.error("Failed to process image: {}", path, e);
            }
        }
    }

    private static int getClassIndex(String[] possibleAnswers, Path path) {
        String fullName = (path.getParent() != null
                ? path.getParent().getFileName().toString() + "_" + path.getFileName().toString()
                : path.getFileName().toString()).toLowerCase();

        int classIndex = -1;
        int bestMatchLength = -1;
        for (int j = 0; j < possibleAnswers.length; j++) {
            String candidate = possibleAnswers[j].toLowerCase();
            if (fullName.contains(candidate) && candidate.length() > bestMatchLength) {
                classIndex = j;
                bestMatchLength = candidate.length();
            }
        }
        return classIndex;
    }

    public void saveSamples(String outDir, String[] names, int count) {
        File dir = new File(outDir);
        dir.mkdirs();
        Random r = new Random();
        int cells = gridSize * gridSize;
        int fpc = outputs[0].length / cells;

        for (int n = 0; n < Math.min(count, floatImages.length); n++) {
            int i = r.nextInt(floatImages.length);
            int size = floatImages[i][0].length;

            BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    int rr = Math.round(floatImages[i][0][y][x] * 255f);
                    int gg = Math.round(floatImages[i][1][y][x] * 255f);
                    int bb = Math.round(floatImages[i][2][y][x] * 255f);
                    img.setRGB(x, y, (rr << 16) | (gg << 8) | bb);
                }
            }

            String label = "NOLABEL";
            for (int c = 0; c < cells; c++) {
                if (outputs[i][c * fpc] > 0.5f) {
                    int best = 0;
                    for (int k = 1; k < fpc - 5; k++) {
                        if (outputs[i][c * fpc + 5 + k] > outputs[i][c * fpc + 5 + best]) best = k;
                    }
                    label = names[best].replace(' ', '_') + "_cell" + c;
                    break;
                }
            }

            // 4x nearest-neighbor upscale so the pixels stay visible
            BufferedImage big = new BufferedImage(size * 4, size * 4, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = big.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
            g.drawImage(img, 0, 0, size * 4, size * 4, null);
            g.dispose();

            try {
                ImageIO.write(big, "png", new File(dir, "sample" + n + "_" + label + ".png"));
            } catch (IOException e) {
                System.err.println("Could not save sample: " + e.getMessage());
            }
        }
        System.out.println("Wrote samples to " + dir.getAbsolutePath());
    } //debug
    public void saveState(String CNNFilePath, String FFNWeightsPath, String FFNBiasPath) {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(CNNFilePath))) {
            dos.writeInt(cnnLayers.length);
            for (ConvolutionalLayer layer : cnnLayers) {
                dos.writeInt(layer.biases.length);
                for (int b = 0; b < layer.biases.length; b++) {
                    dos.writeFloat(layer.biases[b]);
                }

                dos.writeInt(layer.weights.length);       // numFilters
                dos.writeInt(layer.weights[0].length);    // inChannels
                dos.writeInt(layer.weights[0][0].length); // filterHeight (3)
                dos.writeInt(layer.weights[0][0][0].length); // filterWidth (3)

                for (int f = 0; f < layer.weights.length; f++) {
                    for (int c = 0; c < layer.weights[0].length; c++) {
                        for (int y = 0; y < layer.weights[0][0].length; y++) {
                            for (int x = 0; x < layer.weights[0][0][0].length; x++) {
                                dos.writeFloat(layer.weights[f][c][y][x]);
                            }
                        }
                    }
                }
            }

        } catch (IOException e) {
            System.err.println("Failed to save model: " + e.getMessage());
        }

        network.saveState(FFNWeightsPath, FFNBiasPath);

    }

    public void loadState(String CNNFilePath, String FFNWeightsPath, String FFNBiasPath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(CNNFilePath))) {
            int numLayers = dis.readInt();
            if (numLayers != cnnLayers.length) {
                System.err.println("Invalid number of layers: " + numLayers);
            }

            for (ConvolutionalLayer layer : cnnLayers) {
                int numBiases = dis.readInt();
                for (int b = 0; b < numBiases; b++) {
                    layer.biases[b] = dis.readFloat();
                }
                int numFilters = dis.readInt();
                int inChannels = dis.readInt();
                int fHeight = dis.readInt();
                int fWidth = dis.readInt();

                for (int f = 0; f < numFilters; f++) {
                    for (int c = 0; c < inChannels; c++) {
                        for (int y = 0; y < fHeight; y++) {
                            for (int x = 0; x < fWidth; x++) {
                                layer.weights[f][c][y][x] = dis.readFloat();
                            }
                        }
                    }
                }
            }
            network.loadWeights(FFNWeightsPath);
            network.loadBias(FFNBiasPath);

        } catch (IOException e) {
            System.err.println("Failed to load model: " + e.getMessage());
        }
    }
}