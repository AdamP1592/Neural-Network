#ifndef ACTIVATION_FUNCS
#define ACTIVATION_FUNCS
#include <math.h>

struct ActivationResult {
    double activatedValue;
    double derivative;
};
ActivationResult relu(double value);

ActivationResult leakyRelu(double value);

ActivationResult tanH(double value);


#endif