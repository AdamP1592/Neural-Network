#include "NeuralNetworkAPI.h"
#include "network.h"  // Your C++ NeuralNetwork framework header

// Helper function to convert a C-style array to a std::vector<int>
std::vector<int> arrayToVectorInt(const int* arr, int length) {
    return std::vector<int>(arr, arr + length);
}

// Helper function to convert a C-style array to a std::vector<double>
std::vector<double> arrayToVectorDouble(const double* arr, int length) {
    return std::vector<double>(arr, arr + length);
}

extern "C" {

NeuralNetworkHandle createNeuralNetwork() {
    // Allocate a new instance on the heap.
    return new NeuralNetwork();
}

void destroyNeuralNetwork(NeuralNetworkHandle nn) {
    if (nn) {
        delete static_cast<NeuralNetwork*>(nn);
    }
}

void setupNetwork(NeuralNetworkHandle nn, const int* structure, int length) {
    if (!nn || !structure || length <= 0)
        return;
    NeuralNetwork* net = static_cast<NeuralNetwork*>(nn);
    std::vector<int> structVec = arrayToVectorInt(structure, length);
    net->setupNetwork(structVec);
}

std::vector<double> forwardPass(NeuralNetworkHandle nn, const double* inputValues, int numInputs) {
    std::vector<double> inputs;
    if (!nn || !inputValues || numInputs <= 0)
        return inputs;
    NeuralNetwork* net = static_cast<NeuralNetwork*>(nn);
    inputs = arrayToVectorDouble(inputValues, numInputs);
    return net->forwardPass(inputs);
}
void backPropagateRMS(NeuralNetworkHandle nn, const double* expectedValues, int numExpected) {
    if (!nn || !expectedValues || numExpected <= 0)
        return;
    NeuralNetwork* net = static_cast<NeuralNetwork*>(nn);
    std::vector<double> expected = arrayToVectorDouble(expectedValues, numExpected);
    net->backPropagateRMS(expected);
}

void backPropagate(NeuralNetworkHandle nn, const double* expectedValues, int numExpected) {
    if (!nn || !expectedValues || numExpected <= 0)
        return;
    NeuralNetwork* net = static_cast<NeuralNetwork*>(nn);
    std::vector<double> expected = arrayToVectorDouble(expectedValues, numExpected);
    net->backPropagate(expected);
}

} // extern "C"
