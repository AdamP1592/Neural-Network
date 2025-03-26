# Custom Neural Network Framework

This framework implements a feedforward neural network built from scratch in C++. It supports multiple layers, custom activation functions, and adaptive gradient descent using RMSProp. The code is organized into modules for activation functions, neurons, layers, and the overall network. Below, you’ll find an overview of the framework and key equations formatted using $$ for block equations and $ for inline equations.

---

## Table of Contents

1. [Overview](#overview)
2. [Activation Functions](#activation-functions)
3. [Neuron Module](#neuron-module)
4. [Layer Module](#layer-module)
5. [Network Module](#network-module)
6. [Training and Backpropagation](#training-and-backpropagation)
7. [Usage and Extensions](#usage-and-extensions)

---

## Overview

This framework is a fully connected feedforward neural network (multilayer perceptron) configurable with an arbitrary number of layers. It supports both standard gradient descent and an adaptive optimizer using RMSProp. The implementation is done in C++ without external numerical libraries, making it an excellent tool for learning the fundamentals of neural network training.

---

## Activation Functions

The framework supports several activation functions. For example:

- **ReLU (Rectified Linear Unit):**  
  - Function: $ReLU(x)=\max(0,x)$  
  - Derivative: $dReLU(x)=\begin{cases}0 & \text{if } x<0\\1 & \text{if } x\ge0\end{cases}$

- **Leaky ReLU:**  
  - Function: $LeakyReLU(x)=\begin{cases}x & \text{if } x\ge0\\0.01\times x & \text{if } x<0\end{cases}$  
  - Derivative: $dLeakyReLU(x)=\begin{cases}1 & \text{if } x\ge0\\0.01 & \text{if } x<0\end{cases}$

- **tanh (Hyperbolic Tangent):**  
  - Function: $tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$  
  - Derivative: $dtanh(x)=1-tanh^2(x)$  
    Alternatively, it can be computed as:  
    $dtanh(x)=\frac{1}{\cosh^2(x)}$

Each activation function returns both the activated value and its derivative for use during backpropagation.

---

## Neuron Module

The Neuron module handles individual neurons. Key points include:

- **Inputs and Weights:**  
  Each neuron stores a list of weights corresponding to its inputs and a bias value.

- **Activation:**  
  The neuron computes:

$$
sum = bias + \sum_{i=1}^{n}(weight_i \times input_i)
$$

  The sum is then passed through the chosen activation function (e.g., Leaky ReLU or tanh).

- **Backpropagation:**  
  - For output neurons:  

$$
\delta = (activation - target) \times dactivation
$$

  - For hidden neurons:  

$$
\delta = \left( \sum(weight_{next} \times \delta_{next}) \right) \times dactivation
$$

- **RMSProp Integration:**  
  Each neuron maintains a vector of historic gradients. The weight update is computed using:

$$
adjusted\_learning\_rate = \frac{base\_learning\_rate}{\sqrt{historic\_gradient} + \epsilon}
$$

$$
weight = weight - adjusted\_learning\_rate \times (\delta \times input)
$$

  The historic gradient is updated as:

$$
historic\_gradient = (rms\_decay \times historic\_gradient) + (1 - rms\_decay) \times (\delta \times input)^2
$$

---

## Layer Module

Layers group neurons together. Each layer:

- Contains a vector of neurons.
- Provides a method to activate all neurons in the layer.
- Connects neurons from the previous layer by initializing weights and historic gradients appropriately.
- Ensures that the dimensionality of inputs and weights remains consistent.

For example:
- The input layer holds raw data.
- Hidden and output layers perform the weighted sum, apply the activation function, and compute derivatives for backpropagation.

---

## Network Module

The Network module organizes layers and provides methods for:

- **Forward Pass:**  
  Propagating input data through the network layer by layer. For each non-input layer, every neuron computes:

$$
activated\_value = activation\_function\left(bias + \sum_{i}(weight_i \times previous\_layer\_activation_i)\right)
$$

- **Backpropagation:**  
  - For output neurons, compute the error:  

$$
\delta = (activation - target) \times dactivation
$$

  - For hidden layers, errors are propagated backward and used to update each neuron’s delta.
  - Weight updates are performed using the RMSProp method described above.

- **Training:**  
  The training loop processes the dataset, performs forward passes, computes backpropagation errors, and updates weights accordingly.

---

## Training and Backpropagation

Key equations used during training include:

- **Forward Pass Calculation:**  

$$
neuron\_output = activation\_function\left(bias + \sum(weight_i \times input_i)\right)
$$

- **Error at Output Neuron:**  

$$
\delta = (neuron\_output - target) \times dactivation
$$

- **Weight Update (RMSProp):**  

$$
current\_gradient = \delta \times input\_activation
$$

$$
adjusted\_learning\_rate = \frac{learning\_rate}{\sqrt{historic\_gradient} + \epsilon}
$$

$$
weight = weight - adjusted\_learning\_rate \times current\_gradient
$$

  Update historic gradient:

$$
historic\_gradient = (rms\_decay \times historic\_gradient) + ((1 - rms\_decay) \times current\_gradient^2)
$$

These equations drive the training process to minimize the loss function by adjusting the network's weights.

---

## Usage and Extensions

- **Customization:**  
  Modify the network structure by changing the structure vector (e.g., [input_size, hidden1, hidden2, output_size]).

- **Activation Function Options:**  
  Switch between activation functions (ReLU, Leaky ReLU, tanh) to explore their impact on performance.

- **Batch Training and Momentum:**  
  Although the current implementation updates weights per training example, it can be extended to support batch training. Momentum can also be integrated to accelerate convergence.

- **Future Extensions:**  
  - Add convolutional layers for image processing tasks.
  - Implement dropout or batch normalization for improved generalization.
  - Expand optimizer options to include methods like Adam, which combines RMSProp with momentum.

---

This framework is designed for educational and experimental purposes, allowing you to explore neural network training from the ground up. Feel free to modify and extend the code to meet your research or learning objectives.

Happy coding and experimenting!
