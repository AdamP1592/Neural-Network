#include "NeuralNetworkAPI.h"
#include <iostream>
#ifndef BUILD_DLL

int main(){
    NeuralNetworkHandle nn = createNeuralNetwork();

    int* structure = new int[3];
    structure[0] = 3;
    structure[1] = 15;
    structure[2] = 2;

    double inputValues[] = {0.1, 0.5, 0.8};
    double expectedOutputs[] = {0.5, 0};

    double* valuesPointer = inputValues;

    double* outputBuffer = new double[2];
    
    
    setupNetwork(nn, structure, 3);
    for(int i = 0; i < 50; i++){
        forwardPass(nn, inputValues, outputBuffer, 3, 2);
        backPropagateRMS(nn, expectedOutputs, 2);
        printNetwork(nn);
    
    }
    NeuralNetworkHandle newNN = copy(nn);

    //ensures network copy doesnt just reference old network pointer
    backPropagateRMS(nn, expectedOutputs, 2);
    forwardPass(nn, inputValues, outputBuffer, 3, 2);

    std::cout << "base network" << std::endl;
    printNetwork(nn);
    std::cout << "copy network" << std::endl;
    printNetwork(newNN);

    destroyNeuralNetwork(nn);

}
#endif