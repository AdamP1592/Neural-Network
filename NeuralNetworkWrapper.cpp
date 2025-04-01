// NeuralNetworkWrapper.cpp

#include "NeuralNetworkWrapper.h"  // Generated JNI header from javac -h .
#include "NeuralNetworkAPI.h"      // Your existing API header
#include <iostream>
#include "logger.h"

// Use extern "C" to disable C++ name mangling.
extern "C" {

// JNI wrapper for: public native long createNeuralNetwork();
JNIEXPORT jlong JNICALL Java_neuralNetwork_NeuralNetworkWrapper_createNeuralNetwork(JNIEnv* env, jobject obj) {
    std::cout << "Create Network Called" << std::endl;
    // Call the C API function and cast the returned pointer to jlong.
    NeuralNetworkHandle handle = createNeuralNetwork();
    return (jlong)handle;
}

// JNI wrapper for: public native void destroyNeuralNetwork(long nn);
JNIEXPORT void JNICALL Java_neuralNetwork_NeuralNetworkWrapper_destroyNeuralNetwork(JNIEnv* env, jobject obj, jlong nn) {
    destroyNeuralNetwork((NeuralNetworkHandle)nn);
}
// JNI wrapper for: public native long copyNetwork(long nn);
JNIEXPORT jlong JNICALL Java_neuralNetwork_NeuralNetworkWrapper_copyNetwork(JNIEnv* env, jobject obj, jlong nn) {
    NeuralNetworkHandle newNNHandle = copy((NeuralNetworkHandle) nn);
    return (jlong) newNNHandle;
}

// JNI wrapper for: public native void setUpNetwork(long nn, int[] structure, int length);
JNIEXPORT void JNICALL Java_neuralNetwork_NeuralNetworkWrapper_setUpNetwork(JNIEnv* env, jobject obj, jlong nn, jintArray structure, jint length) {
    // Get the pointer to the array elements.
    jint* elements = env->GetIntArrayElements(structure, NULL);
    if (elements == NULL) {
        // If unable to obtain the array, return early (an exception will be pending).
        return;
    }

    // Call the C API function.
    setupNetwork((NeuralNetworkHandle)nn, elements, length);
    // Release the array elements back to the JVM.
    env->ReleaseIntArrayElements(structure, elements, 0);
}

// JNI wrapper for: public native void forwardPass(long nn, double[] inputValues, double[] outputBuffer, int numInputs, int numOutputs);
JNIEXPORT void JNICALL Java_neuralNetwork_NeuralNetworkWrapper_forwardPass(JNIEnv* env, jobject obj, jlong nn, jdoubleArray inputValues, jdoubleArray outputBuffer, jint numInputs, jint numOutputs) {
    // Get the input values array.
    jdouble* inputElements = env->GetDoubleArrayElements(inputValues, NULL);
    if (inputElements == NULL) {
        return;
    }
    // Get the output buffer array.
    jdouble* outputElements = env->GetDoubleArrayElements(outputBuffer, NULL);
    if (outputElements == NULL) {
        env->ReleaseDoubleArrayElements(inputValues, inputElements, 0);
        return;
    }
    // Call the C API forward pass function.
    forwardPass((NeuralNetworkHandle)nn, inputElements, outputElements, numInputs, numOutputs);
    // Release the arrays.
    env->ReleaseDoubleArrayElements(inputValues, inputElements, 0);
    env->ReleaseDoubleArrayElements(outputBuffer, outputElements, 0);
}

// JNI wrapper for: public native void backPropagateRMS(long nn, double[] expected, int numExpected);
JNIEXPORT void JNICALL Java_neuralNetwork_NeuralNetworkWrapper_backPropagateRMS(JNIEnv* env, jobject obj, jlong nn, jdoubleArray expected, jint numExpected) {
    jdouble* expectedElements = env->GetDoubleArrayElements(expected, NULL);
    if (expectedElements == NULL) {
        return;
    }
    backPropagateRMS((NeuralNetworkHandle)nn, expectedElements, numExpected);
    env->ReleaseDoubleArrayElements(expected, expectedElements, 0);
}

// JNI wrapper for: public native void backPropagate(long nn, double[] expected, int numExpected);
JNIEXPORT void JNICALL Java_neuralNetwork_NeuralNetworkWrapper_backPropagate(JNIEnv* env, jobject obj, jlong nn, jdoubleArray expected, jint numExpected) {
    jdouble* expectedElements = env->GetDoubleArrayElements(expected, NULL);
    if (expectedElements == NULL) {
        return;
    }
    backPropagate((NeuralNetworkHandle)nn, expectedElements, numExpected);
    env->ReleaseDoubleArrayElements(expected, expectedElements, 0);
}

} // extern "C"
