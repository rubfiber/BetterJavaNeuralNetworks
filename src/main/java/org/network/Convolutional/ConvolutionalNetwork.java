package org.network.Convolutional;

import org.network.FeedForward.Activation;
import org.network.FeedForward.FeedForwardNetwork;
import org.network.util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

import static org.network.util.*;

public class ConvolutionalNetwork {

    private static final Logger log = LoggerFactory.getLogger(ConvolutionalNetwork.class);
    //filter size is 3x3 to avoid 6 nested loops
    int filterSize;
    public int[] padding;
    public int batchSize;

    public ConvolutionalLayer[] cnnLayers;
    FeedForwardNetwork network;
    int[] filters;


    public ConvolutionalNetwork(filterBuilder builder, int[] FFNLayers, int[] filters, int[] inputChannels, int[] stride) {
        this.filterSize = FFNLayers.length;
        this.filters = filters;
        this.batchSize = builder.batchSize;
        this.padding = builder.padding;

        network = new FeedForwardNetwork(FFNLayers);
        network.setHiddenActivations(Activation.RELU);
        network.setOutputActivation(Activation.SOFTMAX);

        cnnLayers = new ConvolutionalLayer[filters.length];
        for (int i = 0; i < filters.length; i++) {
            cnnLayers[i] = new ConvolutionalLayer(inputChannels[i], filters[i], stride[i], this.padding, 3);
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

    public void train(float[][][] image, float[] desiredOutput) {
        float[][][] cnnOutput = forwardPass(image);
        int lastNumFilters = cnnOutput.length;         // depth/channels
        int lastHeight = cnnOutput[0].length;      // height
        int lastWidth = cnnOutput[0][0].length;   // width

        this.lastCnnHeight = lastHeight;
        this.lastCnnWidth = lastWidth;
        this.lastCnnFilters = lastNumFilters;


        float[] flatData = util.flatten3D(cnnOutput);
        float[] flatError1D = network.accumulateGradientsForCnn(flatData, desiredOutput);
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
        for (int epoch = 0; epoch < epochs; epoch++) {
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
        }

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

        public ConvolutionalLayer(int inChannels, int numFilters, int stride, int[] padding, int kernelSize) {
            this.numFilters = numFilters;
            this.stride = stride;
            this.padding = padding;
            this.weights = new float[numFilters][inChannels][3][3];
            this.biases = new float[numFilters];
            this.kernelSize = kernelSize;


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

        public float[][][] slideOverImage(int[][] image) {
            int width = image.length;
            int height = image[0].length;
            int outHeight = (height - 3 + padding[1]) / stride + 1;
            int outWidth = (width - 3 + padding[0]) / stride + 1;

            float[][][] nextLayerInput = new float[weights.length][outHeight][outWidth];

            for (int filter = 0; filter < weights.length; filter++) {
                float bias = biases[filter];
                float[][][] currentWeights = weights[filter];

                int outY = 0;
                for (int y = Math.floorDiv(padding[1], 2); y < height - 1 && outY < outHeight; y += stride) {
                    int outX = 0;
                    for (int x = Math.floorDiv(padding[0], 2); x < width - 1 && outX < outWidth; x += stride) {

                        float sum = 0f;
                        for (int filterX = x - 1; filterX <= x + 1; filterX++) {
                            for (int filterY = y + 1; filterY >= y - 1; filterY--) {

                                float r = (float) (((image[filterX][filterY] >> 16) & 0xFF) / 255.0);
                                float g = (float) (((image[filterX][filterY] >> 8) & 0xFF) / 255.0);
                                float b = (float) ((image[filterX][filterY] & 0xFF) / 255.0);

                                sum += r * currentWeights[0][filterX - x + 1][filterY - y + 1];
                                sum += g * currentWeights[1][filterX - x + 1][filterY - y + 1];
                                sum += b * currentWeights[2][filterX - x + 1][filterY - y + 1];
                            }
                        }

                        float finalPixel = sum + bias;
                        nextLayerInput[filter][outY][outX] = (finalPixel > 0 ? finalPixel : finalPixel * 0.01f);
                        outX++;
                    }
                    outY++;
                }
            }
            return nextLayerInput;
        }

        public float[][][] slideOverFeatureMap(float[][][] map) { //must be padded

            int height = map[0].length;
            int width = map[0][0].length;

            int outHeight = (height - 3 + padding[1]) / stride + 1;
            int outWidth = (width - 3 + padding[0]) / stride + 1;

            float[][][] nextLayerInput = new float[weights.length][outHeight][outWidth];

            for (int filter = 0; filter < weights.length; filter++) {
                float bias = biases[filter];

                int outY = 0;
                for (int y = 1; y < height - 1 && outY < outHeight; y += stride) {
                    int outX = 0;
                    for (int x = 1; x < width - 1 && outX < outWidth; x += stride) {

                        float sum = 0f;
                        for (int c = 0; c < 3; c++) {
                            for (int filterX = x - 1; filterX <= x + 1; filterX++) {
                                for (int filterY = y - 1; filterY <= y + 1; filterY++) {

                                    float val = map[c][filterY][filterX];
                                    sum += val * weights[filter][c][filterY - y + 1][filterX - x + 1];
                                }
                            }
                        }

                        float finalPixel = sum + bias;
                        nextLayerInput[filter][outY][outX] = (finalPixel > 0 ? finalPixel : finalPixel * 0.01f);
                        outX++;
                    }
                    outY++;
                }
            }
            return nextLayerInput;
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

    // Helper wrapper class to keep track of the image and its path together
    private static class ImageFileData {
        BufferedImage image;
        String fileName;

        ImageFileData(BufferedImage image, String fileName) {
            this.image = image;
            this.fileName = fileName;
        }
    }

    public void addData(String dirPath, String[] possibleAnswers, int targetSize, float brightnessRange, float contrastRange, float saturationRange) {
        ImageFileData[] imageFiles;
        Random rand = new Random();

        try {
            imageFiles = Files.walk(Paths.get(dirPath))
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().matches(".*\\.(png|jpg|jpeg|gif|bmp)$"))
                    .map(path -> {
                        try {
                            BufferedImage img = ImageIO.read(path.toFile());
                            if (img == null) return null;

                            BufferedImage resizedImg = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
                            java.awt.Graphics2D g2d = resizedImg.createGraphics();

                            g2d.drawImage(img, 0, 0, targetSize, targetSize, null);
                            g2d.dispose();
                            return new ImageFileData(resizedImg, path.getFileName().toString().toLowerCase());
                        } catch (IOException e) {
                            System.err.println("Failed to read: " + path);
                            return null;
                        }
                    })
                    .filter(Objects::nonNull)
                    .toArray(ImageFileData[]::new);

            if (imageFiles.length == 0) {
                System.out.println("No images found in " + dirPath);
                return;
            }

            System.out.println("Loaded " + imageFiles.length + " images into the array.");

            int firstImageHeight = imageFiles[0].image.getHeight();
            int firstImageWidth = imageFiles[0].image.getWidth();

            // Check if all images match dimensions
            assert Arrays.stream(imageFiles)
                    .allMatch(item -> item.image.getWidth() == firstImageWidth);
            assert Arrays.stream(imageFiles)
                    .allMatch(item -> item.image.getHeight() == firstImageHeight);

            floatImages = new float[imageFiles.length][3][firstImageHeight][firstImageWidth];
            outputs = new float[imageFiles.length][possibleAnswers.length];

            for (int i = 0; i < imageFiles.length; i++) {
                ImageFileData current = imageFiles[i];

                boolean matchFound = false;
                for (int j = 0; j < possibleAnswers.length; j++) {
                    // Check if the filename contains the target string (ignoring case)
                    if (current.fileName.contains(possibleAnswers[j].toLowerCase())) {
                        outputs[i][j] = 1.0f;
                        matchFound = true;
                        break;
                    }
                }

                if (!matchFound) {
                    System.err.println("Warning: File '" + current.fileName + "' does not match any provided answers. Defaulting to all 0s.");
                }

                // Generate random color jitter factors per image
                float bFactor = (rand.nextFloat() * 2f - 1f) * brightnessRange;               // Brightness delta
                float cFactor = 1.0f + (rand.nextFloat() * 2f - 1f) * contrastRange;         // Contrast multiplier
                float sFactor = 1.0f + (rand.nextFloat() * 2f - 1f) * saturationRange;       // Saturation multiplier

                // Process image RGB pixels with augmentation
                for (int y = 0; y < current.image.getHeight(); y++) {
                    for (int x = 0; x < current.image.getWidth(); x++) {
                        int pixel = current.image.getRGB(x, y);

                        // 1. Base RGB extraction [0.0, 1.0]
                        float red   = (float) ((pixel >> 16) & 0xFF) / 255.0f;
                        float green = (float) ((pixel >> 8)  & 0xFF) / 255.0f;
                        float blue  = (float) (pixel         & 0xFF) / 255.0f;

                        // 2. Brightness Shift
                        red   += bFactor;
                        green += bFactor;
                        blue  += bFactor;

                        // 3. Contrast Adjustment
                        red   = (red   - 0.5f) * cFactor + 0.5f;
                        green = (green - 0.5f) * cFactor + 0.5f;
                        blue  = (blue  - 0.5f) * cFactor + 0.5f;

                        // 4. Saturation Adjustment (via Grayscale Luminance)
                        float gray = 0.299f * red + 0.587f * green + 0.114f * blue;
                        red   = gray + (red   - gray) * sFactor;
                        green = gray + (green - gray) * sFactor;
                        blue  = gray + (blue  - gray) * sFactor;

                        // 5. Clamp to valid [0.0, 1.0] range
                        floatImages[i][0][y][x] = Math.clamp(red, 0.0f, 1.0f);
                        floatImages[i][1][y][x] = Math.clamp(green, 0.0f, 1.0f);
                        floatImages[i][2][y][x] = Math.clamp(blue, 0.0f, 1.0f);
                    }
                }
            }
        } catch (IOException e) {
            log.error("error: ", e);
        }
    }
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
            int numLayers =  dis.readInt();
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
                                    // Overwrite the random initial weights with the trained ones
                                    layer.weights[f][c][y][x] = dis.readFloat();
                                }
                            }
                        }
                    }
                }

        } catch (IOException e) {
            System.err.println("Failed to load model: " + e.getMessage());
        }

        network.loadBias(FFNBiasPath);
        network.loadWeights(FFNWeightsPath);
    }
}