#include "activation_functions.h"

ActivationResult relu(double value) {
    ActivationResult result;
    if (value < 0) {
        result.activatedValue = 0;
        result.derivative = 0;
    } else {
        result.activatedValue = value;
        result.derivative = 1;
    }
    return result;
}
ActivationResult leakyRelu(double value){
    ActivationResult result;
    if (value < 0) {
        
        result.activatedValue = 0.01 * value;
        result.derivative = 0.01;
    } else {
        result.activatedValue = value;
        result.derivative = 1;
    }
    return result;
}



