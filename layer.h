#include "neuron.h"
#include "activation_functions.h"
#include <vector>
#include <functional>
#include <random>

struct Layer{
    //simple layer storage for separation of logic
    // since some activation functions work at the layer scope
    std::vector<Neuron> layer;
    std::function<ActivationResult(double)> activationFunc = leakyRelu;

    int size;

    Layer(int numNeurons, bool isOutput = false){
        size = numNeurons;
        for(int i = 0; i < numNeurons; i++){
            Neuron n(0.1, int(isOutput));
            layer.push_back(n);
        }
    }
    void activate(){
        for(int i = 0; i < size; i++){
            layer[i].activate();
            layer[i].delta = 0;
        }
    }
    void setActivation(const std::string& functionName) {
        if(functionName == "relu") {
            activationFunc = relu;
        } else if(functionName == "tanh") {
            activationFunc = tanH;
        } else {
            activationFunc = leakyRelu;
        }
        for(int i = 0; i < layer.size(); i++){
            layer[i].activationFunc = activationFunc;
        }
    }
    
    void setupReferences(std::vector<std::reference_wrapper<Neuron>> prevLayerNeuronReferences){
        std::random_device rd;
        std::mt19937 gen(rd());

        // Create a uniform real distribution between with a standard deviation
        double standardDev = std::sqrt(2.0/prevLayerNeuronReferences.size());
        std::normal_distribution<double> dis(0.0, standardDev);
        for(int i = 0; i < size; i++){
            
            for(int j = 0; j < prevLayerNeuronReferences.size(); j++){
                layer[i].weights.push_back(dis(gen));
                layer[i].historicGradients.push_back(1.0);
            }
            layer[i].input_neurons = prevLayerNeuronReferences;
        }

    }
    Neuron& getConnection(int neuronIndex){
        return layer[neuronIndex];
    }

    

};