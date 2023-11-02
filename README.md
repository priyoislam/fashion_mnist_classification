## NITEX AI Challenge: Sustainable Apparel Classification

### Objective
This project aims to develop an AI solution using the Fashion MNIST dataset to classify sustainable apparel products, aligning with NITEX's vision. The task involves data analysis, model development, incorporating human expertise for accuracy enhancement, and providing comprehensive documentation.


### Observation
Determining whether a specific item of clothing is sustainable or not can depend on various factors, including the materials used, production processes, ethical practices, and the overall environmental impact.

The Fashin/mnist dataset includes different types of clothing items and footwear. Without specific information about the materials and manufacturing processes used for each individual product, it's challenging to definitively label any of these items as sustainable or not.

However, in general, sustainable apparel products are those made from eco-friendly materials such as organic cotton, hemp, bamboo, or recycled fibers. Additionally, sustainable clothing brands often prioritize ethical practices, fair labor, and environmentally friendly manufacturing processes.


### Dataset

[Fashion MNIST Dataset ](https://www.kaggle.com/datasets/zalando-research/fashionmnist/data)


### Install dependencies:

```
$ pip install -r requirements.txt
```


### Running the Model Evaluation
```
$ python evaluate_model.py /path/to/dataset/folder
```


### Model Architecture

The chosen model consists of three layers:

Input Layer: Flattens the 28x28 images into a 1D array.

Hidden Layer: Dense layer with 512 neurons and ReLU activation.

Output Layer: Dense layer with 10 neurons and Softmax activation for multi-class classification.


Reasoning:

**Simplicity**: Strikes a balance between simplicity and effectiveness.

**Proven Performance**: Similar architectures have shown strong results in image classification tasks.

**Quick Training**: Enables efficient experimentation and model tuning.


**More layers doesn't improve performance** - [Exploring Neural Networks with fashion MNIST](https://medium.com/@ipylypenko/exploring-neural-networks-with-fashion-mnist-b0a8214b7b7b)


This architecture is ideal for accurate and efficient classification of products in the Fashion MNIST dataset
