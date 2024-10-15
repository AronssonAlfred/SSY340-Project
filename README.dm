# Getting Started

1. **Create Git repository**
2. **Create skeleton code** for:
   - Training data upload
   - Validation of object detection
   - Training of emotion detection
3. **Improve the training function** so that you can [pause and save]

---

## Object Detection

1. **Research different object detection models** and select a reasonable one
2. **Select validation data** and run a validation test of the selected model:
   - Check for accuracy and speed performance
3. **Begin skeleton code**, preferably in Jupyter
4. **Compare performance** of different models
5. **Check expected interface**:
   - Input size
   - Input normalization (?)
   - Output size
6. **Create a Python script** to output:
   - Images of detected faces
   - Bounding box (bb) data
7. **Extension to Python script**:
   - Should be able to input live camera feed to the network

---

## Emotion Detection

1. **Select training data**, preprocess it, and split it into different datasets of reasonable size
2. **Begin writing skeleton code**
3. **Research model structure**
4. **Iterate training** until good performance
5. **Write Python script** to run the feedforward model

---

## Create the Demo Program

The demo program should have a live camera feed as input. The picture should be resized and entered into object detection, which will output the bounding box placement and the image in an array. The images should then be input to the emotion detection network to output the given emotion. Finally, the original image should be combined with:
- Bounding box coordinates
- Given emotion

The result should be a live camera feed output, showing the emotion on people's faces.