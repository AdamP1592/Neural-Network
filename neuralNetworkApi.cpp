#include "NeuralNetworkAPI.h"
#include "network.h"  // Your C++ NeuralNetwork framework header

#include <iostream>
namespace{
    // Helper function to convert a C-style array to a std::vector<int>
    std::vector<int> arrayToVectorInt(const int* arr, int length) {
        return std::vector<int>(arr, arr + length);
    }

    // Helper function to convert a C-style array to a std::vector<double>
    std::vector<double> arrayToVectorDouble(const double* arr, int length) {
        return std::vector<double>(arr, arr + length);
    }
    double* vectorToArray(const std::vector<double>& vec) {
        double* arr = new double[vec.size()];
        std::copy(vec.begin(), vec.end(), arr);
        return arr;
    }
}
extern "C" {

NeuralNetworkHandle createNeuralNetwork() {
    // Allocate a new instance on the heap.
    return new NeuralNetwork();
}

NeuralNetworkHandle copy(NeuralNetworkHandle nn){
    NeuralNetwork* currentNet = static_cast<NeuralNetwork*>(nn);

    std::vector<int> nnStructure = currentNet->nnStructure;
    NeuralNetwork* newNN = new NeuralNetwork();

    newNN->setupNetwork(nnStructure);
    currentNet->setupCopy(*newNN);
    
    return newNN;
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

void forwardPass(NeuralNetworkHandle nn, const double* inputValues, double* outputBuffer, int numInputs, int numOutputs) {
    if (!nn || !inputValues || numInputs <= 0)
        return;
    NeuralNetwork* net = static_cast<NeuralNetwork*>(nn);

    std::vector<double> inputVector = arrayToVectorDouble(inputValues, numInputs);
    std::vector<double> outputs = net->forwardPass(inputVector);

    
    // Ensure that the outputs vector has the expected number of elements.
    if (outputs.size() != static_cast<size_t>(numOutputs)) {
        // Optionally, handle the error (e.g., log or copy as many as possible).
        return;
    }

    // Copy the output data to the caller's pre-allocated buffer.
    std::copy(outputs.begin(), outputs.end(), outputBuffer);
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

void printNetwork(NeuralNetworkHandle nn){
    NeuralNetwork* net = static_cast<NeuralNetwork*>(nn);
    net->printNetworkDetailed();

}

} // extern "C"
