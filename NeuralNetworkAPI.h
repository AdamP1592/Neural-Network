#ifndef NEURALNETWORKAPI_H
#define NEURALNETWORKAPI_H
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer type to represent a NeuralNetwork instance.
typedef void* NeuralNetworkHandle;

// Create a new neural network instance.
NeuralNetworkHandle createNeuralNetwork();

// Destroy an existing neural network instance.
void destroyNeuralNetwork(NeuralNetworkHandle nn);

// Set up the network with the given structure.
// `structure` is an array of ints, and `length` is its size.
void setupNetwork(NeuralNetworkHandle nn, const int* structure, int length);

// Perform a forward pass with the given input values.
// `inputValues` is an array of doubles with size equal to the number of input neurons.
std::vector<double> forwardPass(NeuralNetworkHandle nn, const double* inputValues, int numInputs);

// Perform backpropagation using RMSProp with the given expected outputs.
// `expectedValues` is an array of doubles with size equal to the number of output neurons.
void backPropagateRMS(NeuralNetworkHandle nn, const double* expectedValues, int numExpected);

// Optionally, you can add functions to call the standard backpropagation if needed.
void backPropagate(NeuralNetworkHandle nn, const double* expectedValues, int numExpected);

#ifdef __cplusplus
}
#endif

#endif // NEURALNETWORKAPI_H
