/**
 * @file NeuralNetwork.h
 * @brief Testing the Computation of Neural Network in Embedded System. [SM32F030R8T6] chip.
 * @author Waleed Ahmed Daud.
 * @Website waleed-daud.github.io
 * @Linkedin https://linkedin/in/waleed-daud-78472b9b
 * @Email waleed.daud@outlook.com
 * @date OCT 21
 */


#include <stdio.h>

#define NUMBER_OF_INPUT_CELLS 784   /// use 28*28 input cells (= number of pixels per MNIST image)
#define NUMBER_OF_OUTPUT_CELLS 10   /// use 10 output cells to model 10 digits (0-9)

#define LEARNING_RATE  1      /// Incremental increase for changing connection weights
#define HIDDEN_UNITS   2           /// set hidden units number.
#define NUMITERATIONS  1        /// number of iterations.


typedef struct OutputCell OutputCell;
typedef struct HiddenCell HiddenCell;
typedef struct OutputLayer OutputLayer;
typedef struct HiddenLayer HiddenLayer;
typedef struct GeneralLayer GeneralLayer;
typedef struct Vector Vector;




/**
 * @brief Core unit of hidden layer of the neural network.
 */

struct HiddenCell{
    double input [NUMBER_OF_INPUT_CELLS];
    double weight[NUMBER_OF_INPUT_CELLS];
    double bias;
    double dWeight1[NUMBER_OF_INPUT_CELLS];
    double dbias1;
    double z1;
    double a1;
    double dz1;
    double da1;
};

/**
 * @brief Core unit of output layer of the neural network.
 */

struct OutputCell{
    double bias;
    double dbias2;
    double z2;
    double a2;
    double dz2;
    double da2;
    double input[HIDDEN_UNITS]; // the size will be dynamically allocated.
    double weight[HIDDEN_UNITS]; // the size will be dynamically allocated.
    double dWeight2[HIDDEN_UNITS]; // the size will be dynamically allocated.
};


/**
 * @brief The single hidden layer of this network.
 */

struct HiddenLayer{
    HiddenCell cell[HIDDEN_UNITS]; // the size will be dynamically allocated.
};


/**
 * @brief The single output layer of this network.
 */

struct OutputLayer{
    OutputCell cell[NUMBER_OF_OUTPUT_CELLS];
};


/**
 * @brief The General layer of this network.
 */

struct GeneralLayer{
    HiddenLayer hidden_layer;
    OutputLayer output_layer;
};


/**
 * @brief Data structure containing defined number of integer values (the output vector contains values for 0-9)
 */

struct Vector{
    int val[NUMBER_OF_OUTPUT_CELLS];
};


/// ######################################### Functions Set ##########################################
Vector getTargetOutput(int targetIndex);
void initLayer(GeneralLayer *Gl);
void setCellInput(GeneralLayer *Gl, MNIST_Image *img);
void forward_Hidden_cell(GeneralLayer *Gl);
void forward_Output_cell(GeneralLayer *Gl);
void Forward_Propagation(GeneralLayer *Gl ,MNIST_Image *img);
double Cost_Function(GeneralLayer *Gl,Vector *target);
void Backward_Propagation(GeneralLayer *Gl,Vector *target);
void Update_Weights(GeneralLayer *Gl);
int getPrediction(GeneralLayer *Gl);
int Prediction(GeneralLayer *Gl,MNIST_Image *img);
