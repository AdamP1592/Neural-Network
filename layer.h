#include "neuron.h"
#include "activation_functions.h"
#include <vector>
#include <functional>
#include <random>

struct Layer{
    //simple layer storage for separation of logic
    // since some activation functions work at the layer scope
    std::vector<Neuron> layer;
    std::string activationName = "leakyRelu";
    std::function<ActivationResult(double)> activationFunc = leakyRelu;
    std::vector<double> totalDeltas;
    double learningRate;

    int batchSize = 0;

    int size;
    bool batch = false;
    bool outputLayer = false;

    Layer(int numNeurons, bool isOutput = false, bool batchTrain = false, double learningRate = 0.001){
        size = numNeurons;
        batch = batchTrain;
        this->learningRate = learningRate;
        outputLayer = isOutput;


        for(int i = 0; i < numNeurons; i++){
            Neuron n(0.1, int(isOutput));
            totalDeltas.push_back(0);
            layer.push_back(n);
        }
    }
    Layer copyLayer(){
        Layer layerCopy = Layer(size, outputLayer, batch, learningRate);
        
        //to ensure its a copy of all the neurons instead of all the references
        for(int i = 0; i < layer.size(); i++){
            //copies neuron i to layerCopy 
            layerCopy.layer[i] = layer[i];
            //wipe all input neurons
            std::vector<std::reference_wrapper<Neuron>> inputNeurons;
            layerCopy.layer[i].input_neurons = inputNeurons;
        }
        layerCopy.totalDeltas = totalDeltas;
        layerCopy.activationFunc = activationFunc;
        return layerCopy;
    }

    void backPropRMS(std::vector<double> expected = {}){
        if(expected.size() != 0 && expected.size() == layer.size()){
            for(int i = 0; i < layer.size(); i++){
                layer[i].backPropagateRMS(learningRate, 0.9, expected[i]);
                totalDeltas[i] += layer[i].delta;

                layer[i].delta = 0.0;
            }
            return;
        }
        for(int i = 0; i < layer.size(); i++){
            layer[i].backPropagateRMS(learningRate);
            totalDeltas[i] += layer[i].delta;

            layer[i].delta = 0.0;
        }
        

    }

    std::vector<double> activate(std::vector<double> inputs = {}){
        std::vector<double> activations;
        if(inputs.size() != 0 && inputs.size() == layer.size()){
            for(int i = 0; i < size; i++){
                layer[i].activationValue = inputs[i];
                layer[i].delta = 0;
            }
            return activations;
        }
        for(int i = 0; i < size; i++){
            activations.push_back(layer[i].activate());
            layer[i].delta = 0;
            
        }
        return activations;
    }
    void setActivation(const std::string& functionName) {
        activationName = functionName;
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
        bool isCopy = false;
        for(int i = 0; i < size; i++){
            if(layer[i].weights.size() != prevLayerNeuronReferences.size()){
                layer[i].weights.clear();
                layer[i].historicGradients.clear();
                for(int j = 0; j < prevLayerNeuronReferences.size(); j++){
                    layer[i].weights.push_back(dis(gen));
                    layer[i].historicGradients.push_back(1.0);
                }
            }else{
                isCopy = true;
            }
            layer[i].input_neurons = prevLayerNeuronReferences;
        }
        if(isCopy==true){
            std::cout << "Layer Copy";
        }
    }
    Neuron& getConnection(int neuronIndex){
        return layer[neuronIndex];
    }

    

};