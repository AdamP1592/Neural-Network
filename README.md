# Neural-Network
This is a neural network simulation written from scratch in c++.

## Installation

```bash
git clone https://github.com/AdamP1592/Neural-Network
cd Neural-Network

```

## Running the program

```bash 
./Nerual-Network.exe
```

# Neural Network Equations (Leaky ReLU)

## Equations

**Forward Pass (Neuron Activation):**

$$
z = b + \sum_{i=1}^{n} w_i \cdot x_i
$$

$$
a = \text{LeakyReLU}(z) =
\begin{cases}
0.01 \cdot z, & \text{if } z < 0 \\
z, & \text{if } z \ge 0
\end{cases}
$$

$$
f'(z) =
\begin{cases}
0.01, & \text{if } z < 0 \\
1, & \text{if } z \ge 0
\end{cases}
$$

**Where:**

- **z**: Net input to a neuron.
- **b**: Bias of the neuron.
- **wᵢ**: Weight of the connection from the i-th input neuron.
- **xᵢ**: Activation of the i-th input neuron.
- **n**: Total number of input neurons for the current neuron.
- **a**: Activation of the neuron after applying Leaky ReLU.
- **f'(z)**: Derivative of the Leaky ReLU function at z.

**Backpropagation:**

_For an Output Neuron:_

$$
\delta_{\text{output}} = (a - y) \cdot f'(z)
$$

_For a Hidden Neuron:_

$$
\delta_{\text{hidden}} = f'(z_{\text{hidden}}) \cdot \sum_{k=1}^{m} \left( w_{hk} \cdot \delta_k \right)
$$

**Weight and Bias Updates (Gradient Descent):**

$$
w_i \leftarrow w_i - \eta \cdot \delta \cdot x_i
$$

$$
b \leftarrow b - \eta \cdot \delta
$$

**Where**

- **y**: Target output for an output neuron.
- **δ (delta)**: Error term computed during backpropagation.
- **m**: Number of neurons in the next layer (used for hidden neurons).
- **wₕₖ**: Weight connecting a hidden neuron to the k-th neuron in the next layer.
- **δₖ**: Error term of the k-th neuron in the next layer.
- **η (eta)**: Learning rate used during gradient descent.
