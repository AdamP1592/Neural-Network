#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include "activation_functions.h"

class Neuron{
    public:
        double bias;

        double activationValue = 0.0;
        double derivative = 0.0;

        std::vector<double> weights;
        std::vector<Neuron*> input_neurons;

        Neuron(double bias){
            setBias(bias);
        }
        void set(std::vector<Neuron*> prevLayerNeurons, std::vector<double> prevLayerWeights){
            input_neurons = prevLayerNeurons;
            weights = prevLayerWeights;
        }

        void append(Neuron* input_neuron, double weight){
            input_neurons.push_back(input_neuron);

        }

        void printWeights(){
            std::cout << "Weights:";
            for(double weight : weights){
                std::cout << weight << " ";
            }
            std::cout << std::endl;
        }
        double activate(const std::vector<double> inputs){
            if(inputs.size() != weights.size()){
                std::cerr << "Error: input size (" << inputs.size()
                << ") does not match weights size (" << weights.size() << ").\n";
                activationValue = 0;
                return 0.0;
            }
            double sum = bias;
            for (size_t i = 0; i < weights.size(); i++){
                sum += weights[i] * inputs[i];
            }
            ActivationResult reluValues = relu(sum);

            activationValue = reluValues.activatedValue;
            derivative = reluValues.derivative;

            return activationValue;
        }
    private:
        void setWeights(std::vector<double> defaultWeights){
            weights = defaultWeights;
        }
        void setBias(double defaultBias){
            bias = defaultBias;
        }

};


#endif