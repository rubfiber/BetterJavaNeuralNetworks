# Java Neural Nets

These are neural networks written from scratch in Java. Currently, this library supports two architectures:
1. **Feedforward Neural Networks (FFN)** (`org.network.FeedForward.FeedForwardNetwork`)
2. **Convolutional Neural Networks (CNN)** (`org.network.Convolutional.ConvolutionalNetwork`) — supporting both standard image classification and YOLO-style grid object detection & localization.

---

## Installation

You can clone this repository with this command:
```bash
git clone https://github.com/rubfiber/BetterJavaNeuralNetworks/
```

---

## 1. Feedforward Network (`FeedForwardNetwork`)

You can use the `org.network.FeedForward.FeedForwardNetwork` class to create a standard multi-layer perceptron.

### Initialization
Format the constructor by passing an `int[]` array representing the neuron count in each layer (from input layer to output layer):

```java
int[] layers = {784, 128, 64, 10}; // Input: 784, Hidden: 128 -> 64, Output: 10
FeedForwardNetwork network = new FeedForwardNetwork(layers);
```
*(Note: If using an older version of the API, the constructor took `(inputSize, hiddenLayerWidth, hiddenLayerDepth, outputSize)`, which has now been replaced with the flexible `int[] layers` array).*

You can also customize layer activations using `Activation` (`SIGMOID`, `TANH`, `RELU`, `LEAKY_RELU`, `SOFTMAX`, `LINEAR`):
```java
network.fillHiddenActivations(Activation.LEAKY_RELU);
network.fillOutputActivation(Activation.SOFTMAX);
```

### Predictions
Use the `network.predict(float[] input)` method, where `input` is a 1D float array matching your input layer size (`layers[0]`):
```java
float[] prediction = network.predict(inputArray);
```

### Training
Using `network.trainDataset(Dataset dataset, int epochs, int batchSize, float learningRate)`, you can train the network using mini-batch gradient descent:
- `dataset`: A `Dataset` object holding your input data and target output data, both in `float[][]` format (`[sampleIndex][featureIndex]`).
- `epochs`: The number of full training loops through the dataset. Too few results in underfitting; too many can cause overfitting.
- `batchSize`: The number of samples processed before weights are updated in mini-batch gradient descent.
- `learningRate`: Controls the step size during gradient updates.

### Saving & Loading State
The network allows you to save its weights and biases into binary (`.bin`) files.

**Saving:**
```java
network.saveState("weights.bin", "bias.bin");
```
- `weightFilePath`: Path to the `.bin` file where weights will be stored.
- `biasFilePath`: Path to the `.bin` file where biases will be stored.

**Loading:**
```java
network.loadWeights("weights.bin");
network.loadBias("bias.bin");
```

---

## 2. Convolutional Neural Network (`ConvolutionalNetwork`)

You can use the `org.network.Convolutional.ConvolutionalNetwork` class for image classification, object localization, and real-time YOLO-style grid detection. Under the hood, it uses `im2col` matrix multiplication, `3x3` convolutional kernels with Leaky ReLU, max pooling, and a connected `FeedForwardNetwork` head.

### Initialization
First, create a `filterBuilder` to specify padding, batch size, and kernel size (`3`), and then initialize the `ConvolutionalNetwork`:

```java
int[] cnnFilters = {16, 32, 64};       // Output filters per convolutional layer
int[] cnnInputChannels = {3, 16, 32};  // Input channels per layer (3 for RGB)
int[] cnnStrides = {2, 2, 2};          // Stride per convolutional layer
int[] padding = {2, 2};                // Height and width zero-padding

int[][] dummyInputSize = new int[32][32];
ConvolutionalNetwork.filterBuilder builder = new ConvolutionalNetwork.filterBuilder(
        dummyInputSize, padding, 32, 3
);

// For YOLO-style 4x4 grid detection across 33 classes:
int gridSize = 4;
int featuresPerCell = 1 + 4 + 33; // 1 Confidence + 4 Box (x, y, w, h) + 33 Classes = 38
int totalOutputs = (gridSize * gridSize) * featuresPerCell; // 608 Output Neurons

int[] ffnLayers = {2304, 256, totalOutputs}; // Flattened CNN output -> Hidden -> Output

ConvolutionalNetwork cnn = new ConvolutionalNetwork(
        builder, ffnLayers, cnnFilters, cnnInputChannels, cnnStrides
);
cnn.network.fillOutputActivation(Activation.SIGMOID);
cnn.network.featuresPerCell = featuresPerCell;
```

### Loading & Augmenting Image Data
`ConvolutionalNetwork` includes two built-in dataset loaders with automatic color jittering (brightness, contrast, saturation), background compositing, and random spatial shifts:

1. **Standard Classification / Center Shifting (`addData`):**
   ```java
   cnn.addData("src/main/resources/train", labels, 64, 0.2f, 0.3f, 0.4f, 0.4f, Color.WHITE);
   ```
2. **YOLO Grid Detection & Localization (`addDataGrids`):**
   Splits the image into a `gridSize x gridSize` grid, keys out white backgrounds, composites objects onto synthetic/real backgrounds at random scales and positions, and encodes `[confidence, relX, relY, width, height, ...classes]` into the responsible grid cell:
   ```java
   cnn.addDataGrids("src/main/resources/train", labels, 64, 0.2f, 0.3f, 0.4f, 0.4f, Color.WHITE);
   ```
   - `dirPath`: Root directory containing training images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`).
   - `possibleAnswers`: `String[]` array of class names matched against image filenames/folders.
   - `targetSize`: Target width and height to resize images to (e.g., `64` for `64x64`).
   - `brightnessRange`, `contrastRange`, `saturationRange`: Augmentation intensity factors.
   - `shiftRange`: Maximum spatial shift ratio.

*(Optional)* You can export augmented visual samples to verify labels and bounding placements before training:
```java
cnn.saveSamples("Data/", labels, 10);
```

### Training
Train the convolutional network using `trainDataset`:
```java
cnn.trainDataset(cnn.floatImages, cnn.outputs, 0.01f, 32, 25);
```
- `imageInputs`: `float[][][][]` array formatted as `[sampleIndex][channel][y][x]` (RGB normalized `0.0f` to `1.0f`).
- `desiredOutputs`: `float[][]` array of target outputs (`[sampleIndex][outputIndex]`).
- `learningRate`: Step size for both FFN and CNN filter/bias updates.
- `batchSize`: Number of images per mini-batch before calling `applyBatch`.
- `epochs`: Total training passes over the dataset. When `useYoloLoss` is enabled (`true` by default), diagnostic metrics (`loss`, `conf@object`, `conf@empty`, `cell` accuracy, `class` accuracy, `p(true class)`) are printed after each epoch.

### Predictions
Pass a 3D float array `float[3][targetSize][targetSize]` (normalized RGB values between `0.0f` and `1.0f`) to `cnn.predict()`:
```java
float[] prediction = cnn.predict(floatImageArray);
```

### Saving & Loading CNN State
Because a `ConvolutionalNetwork` contains both 4D convolutional filter banks and a `FeedForwardNetwork` head, saving and loading requires three `.bin` paths:

**Saving:**
```java
cnn.saveState(
    "CNNState.bin",   // Convolutional filters and biases
    "FFNWeights.bin", // Feedforward head weights
    "FFNBias.bin"     // Feedforward head biases
);
```

**Loading:**
```java
cnn.loadState(
    "CNNState.bin",
    "FFNWeights.bin",
    "FFNBias.bin"
);
```
## Try it out
### Linux X64 only, so far (AppImage)
Requirements: An X86_64 machine with Linux installed. \
Steps to begin:
1. Download `FruitDetector-Linux-x64.tar.gz` from the latest release.
2. Unpack it and make the launcher executable using the commands:
```bash
 tar xzf FruitDetector-Linux-x64.tar.gz
 chmod +x FruitDetector/bin/FruitDetector
```
3. Run it from a terminal (so you can see any error messages):
```bash   
./FruitDetector/bin/FruitDetector
```
If there were no issues, a camera feed should show up, with a green box following what the network sees as a fruit.\
Common issues:
- "Permission denied" on start: run `chmod +x FruitDetector/bin/FruitDetector FruitDetector/lib/runtime/bin/*`. 
- Camera won't open: try adding yourself to the video group with `sudo usermod -aG video $USER`, then log out and back in. If you have several cameras, the app tries the C270 first on `/dev/video2`, then the default camera, then any other camera it can open.
- Missing .so error on a minimal Debian/Ubuntu: run the command `sudo apt install libxtst6 libxrender1 libxi6 libfreetype6 libfontconfig1 libasound2` in the terminal
- Windows/macOS: not packaged yet. The build script can make packages on those systems with `./gradlew jpackage` (JDK 25 required).
### Additional things to know
  It knows 33 fruits and vegetables: apple, Granny Smith apple, apricot, avocado, banana, blueberry, cactus fruit, cantaloupe, cherry, clementine, corn, cucumber, blue grape, kiwi, lemon, lime, mango, white onion, orange, papaya, passion fruit, peach, pear, green pepper, red pepper, pineapple, plum, pomegranate, potato, raspberry, strawberry, tomato and watermelon.\
  This is a convolutional network written in Java from scratch, so expect wrong predictions, especially if the background is cluttered\
  The trained model is committed in `src/main/resources/model/` (`CNNState.bin`, `FFNWeights.bin`, `FFNBias.bin`) and is bundled into the app. To retrain, put images in `src/main/resources/train` (each file name must contain the fruit/class name) and run `./gradlew trainCNN`. That overwrites the three files above.