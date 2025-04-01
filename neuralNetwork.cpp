#include "network.h"
void printNet(NeuralNetwork neuralNetwork){
    for(int l = 0; l < neuralNetwork.layers.size(); l++){
        Layer& layer = neuralNetwork.layers[l];
        std::cout << "Layer" << l << std::endl;
        for(int n = 0; n < layer.size; n++){
            std::cout << "Neuron " << n << " ";
            layer.layer[n].printWeights();
            std::cout << "Bias: " << layer.layer[n].bias;
        }
        std::cout << std::endl;

    }
}
void useCaseExample(){
    //inputs have to be the same size as the first value in structure
    std::vector<double> inputs = {0, 0, 0};
    //expected is the expected value for the output layer
    std::vector<double> expected = {0, 0, 0, 0, 0};
    //asign the structure
    std::vector<int> structure = {3, 5, 5, 8, 5};

    NeuralNetwork neuralNetwork;
    neuralNetwork.setupNetwork(structure);

    for(Layer layer: neuralNetwork.layers){
        //options
        layer.setActivation("relu");
        layer.setActivation("leakyrelu");
        layer.setActivation("tanh");
    }

    neuralNetwork.forwardPass(inputs);
    neuralNetwork.backPropagateRMS(expected);
    //or 
    neuralNetwork.backPropagate(expected);

}
void simpleTest(){
    Logger::isLogging = true;
    std::vector<double> inputs = {1.0, 3.0, 1.5};
    std::vector<double> expected = {0.5, 0};
    std::vector<int> structure;

    std::cout << "Enter layer structure\n" 
    << "Ex: 1, 2, 1, Yields:\n"
    << "\t0\n0\t\t0\n\t0\n";
    
    std::string layerStructure;
    std::getline(std::cin, layerStructure);
    system("clear");
    structure = {3, 5, 2};
    NeuralNetwork neuralNetwork;
    neuralNetwork.learningRate = 0.001;
    neuralNetwork.setupNetwork(structure);
    
    for(int i = 0; i < 50; i++){

        neuralNetwork.forwardPass(inputs);
        neuralNetwork.backPropagateRMS(expected);

    }

    NeuralNetwork nnCopy;
    nnCopy.setupNetwork(structure);

    neuralNetwork.setupCopy(nnCopy);

    
    std::cout << "Copy visualization\n";
    nnCopy.printNetwork();
    std::cout << "Base visualization\n";
    neuralNetwork.printNetwork();

    std::cout << "Copy state visualization\n";
    printNet(nnCopy);
    std::cout << std::endl;
    std::cout << "Base state visualization\n";
    printNet(neuralNetwork);

    //confirm refrence addresses are right
    for(int i = 1; i < neuralNetwork.layers.size(); i++){
        std::cout << "Layer " << i << std::endl;
        for(int j = 0; j < neuralNetwork.layers[i].layer.size(); j++){
            std::cout << "Neuron " << j << " ";
            Neuron &copiedNeuron = nnCopy.layers[i].layer[j];
            Neuron &originalNeuron = neuralNetwork.layers[i].layer[j];

            int copyInputsSize = nnCopy.layers[i].layer[j].input_neurons.size();
            int nnInputsSize = neuralNetwork.layers[i].layer[j].input_neurons.size();
            for(int k = 0; k < nnCopy.layers[i].layer[j].input_neurons.size(); k++){
                

                std::cout << "Address of neuron in original: " << &originalNeuron.input_neurons[k].get() << "\n";
                std::cout << "Address of neuron in copy: " << &copiedNeuron.input_neurons[k].get() << "\n";

            }
            std::cout << copyInputsSize << ", " << nnInputsSize << std::endl;


        }

    }

    
    nnCopy.forwardPass(inputs);
    neuralNetwork.forwardPass(inputs);

    std::cout << "Copy visualization\n";
    nnCopy.printNetwork();
    std::cout << "Base visualization\n";
    neuralNetwork.printNetwork();
}

#ifndef BUILD_DLL
int main(){
    simpleTest();
    return 0;
}
#endif