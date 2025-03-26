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
  - Function: $ \text{ReLU}(x) = \max(0, x) $  
  - Derivative: $ d\text{ReLU}(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \ge 0 \end{cases} $

- **Leaky ReLU:**  
  - Function: $ \text{LeakyReLU}(x) = \begin{cases} x & \text{if } x \ge 0 \\ 0.01 \times x & \text{if } x < 0 \end{cases} $  
  - Derivative: $ d\text{LeakyReLU}(x) = \begin{cases} 1 & \text{if } x \ge 0 \\ 0.01 & \text{if } x < 0 \end{cases} $

- **tanh (Hyperbolic Tangent):**  
  - Function: $ \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $  
  - Derivative: $ d\tanh(x) = 1 - \tanh^2(x) $  
    Alternatively, it can be computed as:  
    $ d\tanh(x) = \frac{1}{\cosh^2(x)} $

Each activation function returns both the activated value and its derivative for use during backpropagation.

---

## Neuron Module

The Neuron module handles individual neurons. Key points include:

- **Inputs and Weights:**  
  Each neuron stores a list of weights corresponding to its inputs and a bias value.

- **Activation:**  
  The neuron computes:

  $$ \text{sum} = \text{bias} + \sum_{i=1}^{n} (\text{weight}_i \times \text{input}_i) $$

  The sum is then passed through the chosen activation function (e.g., Leaky ReLU or tanh).

- **Backpropagation:**  
  - For output neurons:  
    $$ \delta = (\text{activation} - \text{target}) \times d\text{activation} $$  
  - For hidden neurons:  
    $$ \delta = \left( \sum (\text{weight}_{\text{next}} \times \delta_{\text{next}}) \right) \times d\text{activation} $$

- **RMSProp Integration:**  
  Each neuron maintains a vector of historic gradients. The weight update is computed using:

  $$ \text{adjusted\_learning\_rate} = \frac{\text{base\_learning\_rate}}{\sqrt{\text{historic\_gradient}} + \epsilon} $$

  $$ \text{weight}
