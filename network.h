#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <functional>
#include <sstream>
#include <random>
#include <string>
#include <cctype>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <iomanip>

#include "neuron.h"
#include "layer.h"
#include "logger.h"

#include "network.h"
struct NeuralNetwork{
    std::vector<Layer> layers;
    double learningRate = 1.0;
    int step = 0;
    void setupNetwork(std::vector<int>& structure){
        //creates input layer
        Layer inputLayer = Layer(structure[0]);
        //inputLayer.setActivation()
        
        layers.push_back(inputLayer);
        //creates all hidden layers
        for(int i = 1; i < structure.size() ; i++){
            std::vector<std::reference_wrapper<Neuron>> prevLayerNeuronReferences;
            //creates fist layer
            Layer thisLayer = Layer(structure[i], i == structure.size() - 1 ? true: false);
            //building the references for the next layer
            for(int j = 0; j < structure[i-1]; j++){
                Neuron& refNeuron = layers[i-1].getConnection(j);
                prevLayerNeuronReferences.push_back(refNeuron);
            }
            thisLayer.setupReferences(prevLayerNeuronReferences);

            layers.push_back(thisLayer);
        }
    }
    
    void updateLearningRate(){
        learningRate /= 1.0001;
    }

    //set activationValue for the input layer. Iterate all subsequent layers as a standard pass
    std::vector<double> forwardPass(std::vector<double>& inputValues){

        std::vector<double> outputs;

        Logger::log("forwardPass:\n");
        Logger::log("Learning Rate: ");
        Logger::log(std::to_string(learningRate));
        //catch cases for errors
        if(layers.empty()){
            std::cerr << "Error: No layers in the network.\n";
            return outputs;
        }
        if(inputValues.size() != layers[0].size){
            std::cerr << "Error: input size (" << layers[0].size
            << ") does not match network imports size (" << inputValues.size() << ").\n";
            return outputs;
        }
        //set all the input layer activation values to the inputs
        for (int i = 0; i < layers[0].size; i++){
            Neuron& n = layers[0].layer[i];
            n.activationValue = inputValues[i];
            
            //logging activation
            std::ostringstream oss;
            oss << "Neuron: (0, " << i << ") Effective learning rate: " << n.adjustedLearningRate << "Activation:" << inputValues[i];
            Logger::log(oss.str());
        }
        //iterate each non input layer, activate all neurons in the layers
        int size = layers.size();
        for(int i = 1; i < size; i++){
            //get the current layer reference
            Layer& thisLayer = layers[i];

            int layerSize = layers[i].size;
            for(int j = 0; j < layerSize; j++){
                //get current neuron reference
                thisLayer.layer[j].activate();
                Neuron& n = thisLayer.layer[j];

                //logging activation
                std::ostringstream oss;
                oss << "Neuron: (" << i << ", " << j << ") Activation:" << n.activationValue;
                Logger::log(oss.str());
            }
        }

        Layer outputLayer = layers[layers.size() - 2];
        for(int i = 0; i < outputLayer.layer.size(); i++){
            outputs.push_back(outputLayer.layer[i].activationValue);
        }
        return outputs;
    }
    void backPropagate(std::vector<double>& expectedValues){
        //catch case for empty network
        Logger::log("BackProp:");
        std::string expectedValuesString;
        Logger::log("Expected");
        for(int i = 0; i < expectedValues.size(); i++){
            expectedValuesString += " " + std::to_string(expectedValues[i]);
        }
        Logger::log(expectedValuesString + "\n");

        if(layers.empty()){
            std::cerr << "Error: No layers in the network.\n";
            return;
        }
        Layer &outputLayer = layers[layers.size() - 1];
        //catch case for size mismatch
        if(expectedValues.size() != outputLayer.size){
            std::cerr << "Error: expectedValues size (" << outputLayer.size
            << ") does not match network output size (" << expectedValues.size() << ").\n";
            return;
        }

        //back prop call for output layer
        for(int i = 0; i < outputLayer.size; i++){
            Neuron &n = outputLayer.layer[i];
            double err = n.backPropagateRMS(learningRate, 0.9, expectedValues[i]);

            //logging backprop
            std::ostringstream oss;
            oss << "Neuron: (" << layers.size() - 1  << ", " << i << "), NeuronType: " << n.neuronType << " Error: " << err;
            Logger::log(oss.str());
        }
        //skip output layer
        for(int i = layers.size() - 2; i >= 0; i--){
            Layer &curLayer = layers[i];
            for(int j = 0; j < curLayer.size; j++){
                Neuron &n = layers[i].layer[j];
                double err = n.backPropagateRMS(learningRate);

                //logging with string builder
                std::ostringstream oss;
                oss << "Neuron: (" << i  << ", " << j << ") backProp Error:" << err;
                Logger::log(oss.str());
            }

        }
        step++;
        updateLearningRate();

    }

    void backPropagateRMS(std::vector<double>& expectedValues){
        //catch case for empty network
        Logger::log("BackProp:");
        std::string expectedValuesString;
        Logger::log("Expected");
        for(int i = 0; i < expectedValues.size(); i++){
            expectedValuesString += " " + std::to_string(expectedValues[i]);
        }
        Logger::log(expectedValuesString + "\n");

        if(layers.empty()){
            std::cerr << "Error: No layers in the network.\n";
            return;
        }
        Layer &outputLayer = layers[layers.size() - 1];
        //catch case for size mismatch
        if(expectedValues.size() != outputLayer.size){
            std::cerr << "Error: expectedValues size (" << outputLayer.size
            << ") does not match network output size (" << expectedValues.size() << ").\n";
            return;
        }
        //iterate backwards from the last layer to the first hidden layer
        for(int i = layers.size() - 1; i > 0; i--){
            for(int j = 0; j < layers[i].layer.size(); j++){
                Logger::log("Neuron (" + std::to_string(i) + ", " + std::to_string(j) + ")\n");
                layers[i].layer[j].backPropagateRMS(learningRate, 0.9, expectedValues[j]);
            }
        }
        step++;
    }

    void printNetworkDetailed() {
        std::cout << "Neural Network Visualization:\n";
        std::cout << "Base Learning Rate: " << learningRate << "\n";
        int numLayers = layers.size();
        for (int l = 0; l < numLayers; l++){
            std::cout << "Layer " << l << " (" << layers[l].size << " neurons):\n";

            for (int n = 0; n < layers[l].size; n++){
                Neuron& neuron = layers[l].layer[n];
                // Format the numbers with fixed precision
                std::cout << "  Neuron " << std::setw(2) << n << " Neuron type: " << neuron.neuronType 
                        << " | Effective learning rate: " << std::fixed << std::setprecision(4) << neuron.adjustedLearningRate 
                        << " | Activation: " << std::fixed << std::setprecision(4) << neuron.activationValue 
                        << " | Error: " << std::fixed << std::setprecision(4) << neuron.delta
                        << "\n";
            }
            std::cout << std::endl;
        }
    }

    void printExpectedOutputs(const std::vector<double>& expected) {
        std::cout << "Expected Outputs:\n";
        for (size_t i = 0; i < expected.size(); i++){
            std::cout << "  Output " << std::setw(2) << i 
                      << " | Value: " << std::fixed << std::setprecision(4) << expected[i] 
                      << "\n";
        }
        std::cout << std::endl;
    }
    void printNetwork(){
        printNetworkDetailed();
    }

};


#endif