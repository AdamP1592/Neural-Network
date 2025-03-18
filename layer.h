#include "neuron.h"
#include <vector>
#include <functional>
#include <random>

struct Layer{
    std::vector<Neuron> layer;
    int size;

    Layer(int numNeurons){
        size = numNeurons;
        for(int i = 0; i < numNeurons; i++){
            Neuron n(0.0);
            layer.push_back(n);
        }
    }
    void activate(){
        for(int i = 0; i < size; i++){
            layer[i].activate();
        }
    }
    void setupReferences(std::vector<std::reference_wrapper<Neuron>> prevLayerNeuronReferences){
        std::random_device rd;
        std::mt19937 gen(rd());

        // Create a uniform real distribution between -0.5 and 0.5.
        std::uniform_real_distribution<double> dis(-0.5, 0.5);
        for(int i = 0; i < size; i++){
            for(size_t j = 0; j < prevLayerNeuronReferences.size(); j++){
                layer[i].weights.push_back(dis(gen));
            }
            layer[i].input_neurons = prevLayerNeuronReferences;
        }

    }
    Neuron& getConnection(int neuronIndex){
        return layer[neuronIndex];
    }

    

};