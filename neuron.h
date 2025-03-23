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
    double adjustedLearningRate = 0.0;
    double derivative = 0.0;
    double delta = 0.0;
    int neuronType = 0;

    std::vector<double> weights;

    std::vector<double> historicGradients;
    std::vector<std::reference_wrapper<Neuron>> input_neurons;

    // Constructor
    Neuron(double bias, int nType = 0) {
        setBias(bias);
        neuronType = nType;
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
        ActivationResult reluValues = leakyRelu(sum);
        activationValue = reluValues.activatedValue;
        derivative = reluValues.derivative;

        return activationValue;
    }
    double backPropagateOutput(double targetValue, double learningRate){
        double deltaOutput = (activationValue - targetValue) * derivative;

        delta = deltaOutput;

        //pass the current neuron delta to all the connected neurons
        for(int i = 0; i < weights.size(); i++){
            input_neurons[i].get().delta += weights[i] * deltaOutput;
        }
        //update weights
        for(int i = 0; i < weights.size(); i++){

            double inputActivation = input_neurons[i].get().activationValue;
            weights[i] -= learningRate * delta * inputActivation;

        }
        bias -= delta * learningRate;

        return deltaOutput;
    }
    double backPropagate(double learningRate, double targetValue = 0.0){

        if (neuronType == 1){
            double deltaOutput = (activationValue - targetValue) * derivative;
            delta = deltaOutput;
        }
        delta *= derivative;

        for(int i = 0; i < weights.size(); i++){
            input_neurons[i].get().delta += weights[i] * delta;
        }

        for(int i = 0; i < weights.size(); i++){

            double inputActivation = input_neurons[i].get().activationValue;
            double currentGradient =  delta * inputActivation;

            weights[i] -= currentGradient * learningRate;

        }
        bias -= delta * learningRate;
        
        return delta;
    }
    double backPropagateRMS(double learningRate, double rmsDecay = 0.9, double targetValue = 0.0){
        double epsilon = 1e-8;
        //if neuron is an output neuron start the backprop
        
        if (neuronType == 1){
            delta = activationValue - targetValue;
        }
        delta *= derivative;

        for(int i = 0; i < weights.size(); i++){
            input_neurons[i].get().delta += weights[i] * delta;
        }
        for(int i = 0; i < weights.size(); i++){

            double inputActivation = input_neurons[i].get().activationValue;
            double currentGradient =  delta * inputActivation;

            adjustedLearningRate = learningRate / (std::sqrt(historicGradients[i]) + epsilon);

            //clip learning rate at 5 to prevent network explosion as a result of an unused neuron
            //getting activated as a result of new data
            adjustedLearningRate = std::min(adjustedLearningRate, 5.0);

            weights[i] -= (adjustedLearningRate * currentGradient);

            historicGradients[i] = rmsDecay * historicGradients[i] + (1 - rmsDecay) * (currentGradient * currentGradient);

        }
        bias -= delta * learningRate;
        
        return delta;
    }

private:
    void setWeights(const std::vector<double>& defaultWeights) {
        weights = defaultWeights;
        for(int i = 0; i < defaultWeights.size(); i++){
            historicGradients.push_back(0.0);
        }
    }
    void setBias(double defaultBias) {
        bias = defaultBias;
    }
};

#endif // NEURON_H
