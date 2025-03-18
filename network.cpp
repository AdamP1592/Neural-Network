
#include <vector>
#include <functional>
#include <sstream>
#include <random>
#include <string>
#include <cctype>
#include "neuron.h"
#include "layer.h"
struct network{
    std::vector<Layer> layers;

    void setupNetwork(std::vector<int> structure){
        
        Layer inputLayer = Layer(structure[0]);
        layers.push_back(inputLayer);
        
        for(int i = 1; i < structure.size(); i++){
            std::vector<std::reference_wrapper<Neuron>> prevLayerNeuronReferences;
            std::vector<double> weights;

            Layer thisLayer = Layer(structure[i]);
            //building the references for the next layer
            for(int j = 0; j < structure[i-1]; j++){
                Neuron& refNeuron = layers[i-1].getConnection(j);
                prevLayerNeuronReferences.push_back(refNeuron);
            }
            thisLayer.setupReferences(prevLayerNeuronReferences);
            

            layers.push_back(thisLayer);
            
        }

    }

    //set activationValue for the input layer. Iterate all subsequent layers as a standard pass
    void forwardPass(std::vector<double>& inputValues){
        //catch cases for errors
        if(layers.empty()){
            std::cerr << "Error: No layers in the network.\n";
            return;
        }
        if(inputValues.size() != layers[0].size){
            std::cerr << "Error: input size (" << layers[0].size
            << ") does not match network imports size (" << inputValues.size() << ").\n";
            return;
        }
        //set all the input layer activation values to the inputs
        for (int i = 0; i < layers[0].size; i++){
            layers[0].layer[i].activationValue = inputValues[i];
        }
        //iterate each non input layer, activate all neurons in the layers
        int size = layers.size();
        for(int i = 1; i < size; i++){
            //get the current layer reference
            Layer& thisLayer = layers[i];

            int layerSize = layers[i].size;
            for(int j = 0; j < layerSize; j++){
                //get current neuron reference
                Neuron& n = thisLayer.layer[j];
                n.activate();
            }
        }
    }

    void printNetwork(){
        int size = layers.size();
        for(int i = 0; i < size; i++){
            int layerSize = layers[i].size;
            Layer& thisLayer = layers[i];
            for(int j = 0; j < layerSize; j++){
                Neuron& n = thisLayer.layer[j];

                std::cout << n.activationValue << " ";
            }
            std::cout << std::endl;
        }
    }

};

std::vector<int> stringToStructure(std::string layerStructure){
    std::vector<int> structure;
    std::string thisInt;
    for(int i = 0; i < layerStructure.length(); i++){
       
        if(isdigit(layerStructure[i])){
            thisInt.push_back(layerStructure[i]);
        }else if(layerStructure[i] == ',' || i == layerStructure.length() - 1){
            if(!thisInt.empty()){
                structure.push_back(std::stoi(thisInt));
                thisInt.clear();
            }
        }
    }
    if(!thisInt.empty()){
        structure.push_back(std::stoi(thisInt));
        thisInt.clear();
    }
    //catch case for any unknown structure; 
    return structure;
}


int main(){
    
    std::vector<double> inputs = {1.0, 3.0, 1.5};
    std::vector<double> weights = {0.5, -0.5, 1.0};
    std::vector<int> structure;

    double bias = 0.1;

    Neuron n(bias);
    
    double output = n.activate();

    std::cout << "Enter layer structure\n" 
    << "Ex: 1, 2, 1, Yields:\n"
    << "\t0\n0\t\t0\n\t0\n";
    
    std::string layerStructure;
    std::getline(std::cin, layerStructure);
    system("clear");

    structure = stringToStructure(layerStructure);
    
    network neuralNetwork;
    neuralNetwork.setupNetwork(structure);

    neuralNetwork.printNetwork();

    neuralNetwork.forwardPass(inputs);

    neuralNetwork.printNetwork();

    std::getline(std::cin, layerStructure);
    ///std::cout << "Feed-forward output: " << output << std::endl;
    return 0;
}