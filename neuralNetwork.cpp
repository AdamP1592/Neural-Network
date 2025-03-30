#include "network.h"
void simpleTest(){
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
        
        for(int l = 0; l < neuralNetwork.layers.size(); l++){
            Layer& layer = neuralNetwork.layers[l];
            std::cout << "Layer" << l << std::endl;
            for(int n = 0; n < layer.size; n++){
                std::cout << "Neuron" << n << " ";
                layer.layer[n].printWeights();
            }
            std::cout << std::endl;

        }
        std::cout << "Forward Pass \n";
        neuralNetwork.forwardPass(inputs);
        std::cout << "Back Pass \n";
        neuralNetwork.backPropagateRMS(expected);


        neuralNetwork.printNetworkDetailed();
        neuralNetwork.printExpectedOutputs(expected);
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

int main(){
    simpleTest();
    return 0;
}