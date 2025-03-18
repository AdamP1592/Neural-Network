#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include <functional>    // For std::reference_wrapper
#include "activation_functions.h"

class Neuron {
public:
    double bias;
    double activationValue = 0.0;
    double derivative = 0.0;
    std::vector<double> weights;
    std::vector<std::reference_wrapper<Neuron>> input_neurons;

    // Constructor
    Neuron(double bias) {
        setBias(bias);
    }

    // Set the input neurons and corresponding weights
    void set(const std::vector<std::reference_wrapper<Neuron>>& prevLayerNeurons, const std::vector<double>& prevLayerWeights) {
        input_neurons = prevLayerNeurons;
        weights = prevLayerWeights;
    }

    // Append a neuron and weight to the input list
    void append(Neuron& input_neuron, double weight) {
        input_neurons.push_back(std::ref(input_neuron));
        weights.push_back(weight);
    }

    void printWeights() {
        std::cout << "Weights: ";
        for (double weight : weights) {
            std::cout << weight << " ";
        }
        std::cout << std::endl;
    }

    // Activate the neuron using stored input neurons and weights.
    double activate() {
        if (input_neurons.size() != weights.size()) {
            std::cerr << "Error: input size (" << input_neurons.size()
                      << ") does not match weights size (" << weights.size() << ").\n";
            activationValue = 0;
            return 0.0;
        }

        double sum = bias;
        for (size_t i = 0; i < weights.size(); i++) {
            // Access the neuron via .get() from the reference wrapper.
            sum += weights[i] * input_neurons[i].get().activationValue;
        }
        ActivationResult reluValues = relu(sum);
        activationValue = reluValues.activatedValue;
        derivative = reluValues.derivative;

        return activationValue;
    }

private:
    void setWeights(const std::vector<double>& defaultWeights) {
        weights = defaultWeights;
    }
    void setBias(double defaultBias) {
        bias = defaultBias;
    }
};

#endif // NEURON_H
