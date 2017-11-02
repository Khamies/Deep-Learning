/**
 * @file NeuralNetwork.c
 * @brief Testing the Computation of Neural Network in Embedded System. [SM32F030R8T6] chip.
 * @author Waleed Ahmed Daud.
 * @Website waleed-daud.github.io
 * @Linkedin https://linkedin/in/waleed-daud-78472b9b
 * @Email waleed.daud@outlook.com
 * @date OCT 21
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "mnist-utils.h"
#include "Neural-Network-v1-NN.h"







/**
 * @details Initialize layer by setting all weights to random values [0-1] and the inputs to zeros.
 */

void initLayer(GeneralLayer *Gl){

    /// initialization of Hidden layer cells.
        int o;
    for ( o=0; o<HIDDEN_UNITS; o++){
        int i;
        for (i=0; i<NUMBER_OF_INPUT_CELLS; i++){

            Gl->hidden_layer.cell[o].input[i]=0;
            Gl->hidden_layer.cell[o].weight[i]=rand()/(double)(RAND_MAX);
            Gl->hidden_layer.cell[o].dWeight1[i]=0;

        }

        Gl->hidden_layer.cell[o].bias = 0;
        Gl->hidden_layer.cell[o].dbias1=0;
        Gl->hidden_layer.cell[o].z1=0;
        Gl->hidden_layer.cell[o].a1=0;
        Gl->hidden_layer.cell[o].dz1=0;
        Gl->hidden_layer.cell[o].da1=0;
    }

    /// initialization of Output layer cells.
    for ( o=0; o<NUMBER_OF_OUTPUT_CELLS; o++){
        int i;
        for (i=0; i<HIDDEN_UNITS; i++){
            Gl->output_layer.cell[o].input[i]=0;
            Gl->output_layer.cell[o].weight[i]=rand()/(double)(RAND_MAX);
            Gl->output_layer.cell[o].dWeight2[i]=0;
        }


        Gl->output_layer.cell[o].bias = 0;
        Gl->output_layer.cell[o].dbias2=0;
        Gl->output_layer.cell[o].z2=0;
        Gl->output_layer.cell[o].a2=0;
        Gl->output_layer.cell[o].dz2=0;
        Gl->output_layer.cell[o].da2=0;
    }

}



void resetLayer(GeneralLayer *Gl)
{
         int o;
    for ( o=0; o<HIDDEN_UNITS; o++)
        {

        Gl->hidden_layer.cell[o].z1=0;
        Gl->hidden_layer.cell[o].a1=0;
        }

    for ( o=0; o<NUMBER_OF_OUTPUT_CELLS; o++)
        {

        Gl->output_layer.cell[o].z2=0;
        Gl->output_layer.cell[o].a2=0;
        }
}

/**
 * @details Creates an input vector of length NUMBER_OF_INPUT_CELLS
 * of a given MNIST image, setting input vector cells to [0,1]
 * based on the pixels of the image.
 * Scalar pixel intensity [=grey-scale] is ignored, only 0 or 1 [=black-white].
 */

void setCellInput(GeneralLayer *Gl, MNIST_Image *img)
{
        int o;
    for ( o=0; o<HIDDEN_UNITS; o++){
        int i;
        for (i=0; i<NUMBER_OF_INPUT_CELLS; i++){
        Gl->hidden_layer.cell[o].input[i] = img->pixel[i] ? 1 : 0;
    }
}
}




/**
 * @details  forward propagation for hidden layer.
 */

void forward_Hidden_cell(GeneralLayer *Gl)
{

     int o;
    for ( o=0; o<HIDDEN_UNITS; o++)
    {

        int i;
        for (i=0; i<NUMBER_OF_INPUT_CELLS; i++)
        {
        Gl->hidden_layer.cell[o].z1+=Gl->hidden_layer.cell[o].input[i] * Gl->hidden_layer.cell[o].weight[i];
        }

        Gl->hidden_layer.cell[o].z1+=Gl->hidden_layer.cell[o].bias;

        Gl->hidden_layer.cell[o].a1=tanh(Gl->hidden_layer.cell[o].z1); /// "tanh" as activation function for hidden units.

    //printf(" z1 is: %lf \n",Gl->hidden_layer.cell[o].z1) ;                                                                         /// to make the output between [0-1].
   // printf(" a1 is: %lf \n",Gl->hidden_layer.cell[o].a1) ;



    }
}

/**
 * @details
 */

void forward_Output_cell(GeneralLayer *Gl)
{
    int o;
    for ( o=0; o<NUMBER_OF_OUTPUT_CELLS; o++)
{
        int i;
        for (i=0; i<HIDDEN_UNITS; i++)
        {
        Gl->output_layer.cell[o].z2+=Gl->hidden_layer.cell[i].a1 * Gl->output_layer.cell[o].weight[i];

        }

        Gl->output_layer.cell[o].z2+=Gl->output_layer.cell[o].bias;

        Gl->output_layer.cell[o].a2= 1/(1+exp(-(Gl->output_layer.cell[o].z2))); /// sigmoid function as activation function

        //printf(" z2 is: %lf \n",Gl->output_layer.cell[o].z2) ;                                                                         /// to make the output between [0-1].
        printf(" a2 is: %lf \n",Gl->output_layer.cell[o].a2) ;
}


}

/**
 * @details
 */


void Forward_Propagation(GeneralLayer *Gl ,MNIST_Image *img)
{
    resetLayer(Gl);    /// reset z1,a1,z2,a2
    setCellInput(Gl, img);
    forward_Hidden_cell(Gl);
    forward_Output_cell(Gl);

}



double Cost_Function(GeneralLayer *Gl,Vector *target)
{
    double cost;
    int o;
    for ( o=0; o<NUMBER_OF_OUTPUT_CELLS; o++)
        {
            //printf("target is: %d \n\n",target->val[o]);
            //printf("a2: %lf \n\n",Gl->output_layer.cell[o].a2);
            cost+= -( ( target->val[o]*log(Gl->output_layer.cell[o].a2)) + ((1-target->val[o])*log(1-(Gl->output_layer.cell[o].a2))));
            //cost+=pow( (target->val[o]- Gl->output_layer.cell[o].a2) ,2);


        }

        printf("cost : %lf \n",cost);

        return cost;

}




void Backward_Propagation(GeneralLayer *Gl,Vector *target)
{
/// ##################  calculate dz2,dw2,db2   ######################.

/// dz2
    int o;
    for ( o=0; o<NUMBER_OF_OUTPUT_CELLS; o++)
        {
            Gl->output_layer.cell[o].dz2=Gl->output_layer.cell[o].a2 - target->val[o];
           // printf(" dz2 %lf \n",Gl->output_layer.cell[o].dz2);
        }

/// dw2
    for ( o=0; o<NUMBER_OF_OUTPUT_CELLS; o++)
        {

         int i;
        for (i=0; i<HIDDEN_UNITS; i++)
        {

            Gl->output_layer.cell[o].dWeight2[i]=Gl->hidden_layer.cell[i].a1 * Gl->output_layer.cell[o].dz2;

        }

        }

/// db2


    for ( o=0; o<NUMBER_OF_OUTPUT_CELLS; o++)
    {
        Gl->output_layer.cell[o].dbias2=Gl->output_layer.cell[o].dz2;

    }

/// ##################  calculating dz1,dw1,db1   ######################.


/// dz1

        int n;
         for ( n=0; o<HIDDEN_UNITS; n++)
        {
            double sum1=0;
            int o;
        for (o=0; o<NUMBER_OF_OUTPUT_CELLS; o++)
            {
                double sum2=0;
                 int m;
                for(m=0;m<HIDDEN_UNITS;m++)
                {
                    sum2+=(Gl->output_layer.cell[o].weight[m])*(Gl->output_layer.cell[o].dz2);
                }
                sum1+=sum2;
            }

            Gl->hidden_layer.cell[n].dz1=sum1*(1-pow(Gl->hidden_layer.cell[n].a1,2));
        }

/// dW1

         for ( o=0; o<HIDDEN_UNITS; o++)
        {

         int i;
        for (i=0; i<NUMBER_OF_INPUT_CELLS; i++)
        {

            Gl->hidden_layer.cell[o].dWeight1[i]=Gl->hidden_layer.cell[o].input[i] * Gl->hidden_layer.cell[o].dz1;
        }

        }

/// db1



    for ( o=0; o<HIDDEN_UNITS; o++)
    {
        Gl->hidden_layer.cell[o].dbias1=Gl->hidden_layer.cell[o].dz1;

    }


}



void Update_Weights(GeneralLayer *Gl)
{

/// update w1
    int o;
    for ( o=0; o<HIDDEN_UNITS; o++)
        {
            int i;
        for (i=0; i<NUMBER_OF_INPUT_CELLS; i++)
        {
            Gl->hidden_layer.cell[o].weight[i]=Gl->hidden_layer.cell[o].weight[i]-(LEARNING_RATE*Gl->hidden_layer.cell[o].dWeight1[i]);
             //printf(" updated weight1 %lf \n",Gl->hidden_layer.cell[o].weight[i]);
        }
/// update b1
         Gl->hidden_layer.cell[o].bias=Gl->hidden_layer.cell[o].bias-(LEARNING_RATE*Gl->hidden_layer.cell[o].dbias1);
         //printf(" updated bias %lf \n",Gl->hidden_layer.cell[o].bias);
        }

/// update w2
    for ( o=0; o<NUMBER_OF_OUTPUT_CELLS; o++)
        {
        int i;
        for (i=0; i<HIDDEN_UNITS; i++)
        {
            Gl->output_layer.cell[o].weight[i]=Gl->output_layer.cell[o].weight[i]-(LEARNING_RATE*Gl->output_layer.cell[o].dWeight2[i]);
            //printf(" updated weight2 %lf \n",Gl->output_layer.cell[o].weight[i]);

        }
/// update b2
         Gl->output_layer.cell[o].bias=Gl->output_layer.cell[o].bias-(LEARNING_RATE*Gl->output_layer.cell[o].dbias2);
         //printf(" updated bias2 %lf \n",Gl->output_layer.cell[o].bias);

        }


}


int getPrediction(GeneralLayer *Gl){

    double maxOut = 0;
    int maxInd = 0;
        int i;
    for ( i=0; i<NUMBER_OF_OUTPUT_CELLS; i++){


   // printf(" Maxout: %lf",Gl->output_layer.cell[i].a2);
        if (Gl->output_layer.cell[i].a2 > maxOut){
            maxOut = Gl->output_layer.cell[i].a2;
            maxInd = i;

        }

    }

    return maxInd;

}


/**
 * @details The output prediction is derived by simply sorting all output values
 * and using the index (=0-9 number) of the highest value as the prediction.
 */



int Prediction(GeneralLayer *Gl,MNIST_Image *img)
{
    Forward_Propagation(Gl,img);
    int predictedNum=getPrediction(Gl);

    return predictedNum;

}




/**
 * @details Returns an output vector with targetIndex set to 1, all others to 0
 */

Vector getTargetOutput(int targetIndex){
    Vector v;
    int i;
    for ( i=0; i<NUMBER_OF_OUTPUT_CELLS; i++){
        v.val[i] = (i==targetIndex) ? 1 : 0;
    }
    return v;
}




void delay(unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}



void export_Weights(GeneralLayer *Gl)
{




 /// export weight1 in hidden cells.
int o;

for(o=0;o<HIDDEN_UNITS;o++)
{
char filename[HIDDEN_UNITS];
sprintf(filename, "weights1_cell%d.txt", o);

FILE *f;
f = fopen(filename, "w+");
int i;
for(i=0;i<NUMBER_OF_INPUT_CELLS;i++)
{

fprintf(f, "%.5g\n",Gl->hidden_layer.cell[o].weight[i]);
}
fclose(f);
}

printf("weights1 have been exported ! \n\n");

///####################################### export bias in hidden cells #####################################


char filename_bias1[]="bias1.txt";

FILE *f = fopen(filename_bias1, "w+");
int i;
for(i=0;i<HIDDEN_UNITS;i++)
{

fprintf(f, "%.5g\n",Gl->hidden_layer.cell[i].bias);
}

fclose(f);

printf("biases1 have been exported \n\n");


/// export weight2 in output cells
for(o=0;o<NUMBER_OF_OUTPUT_CELLS;o++)
{
char filename[NUMBER_OF_OUTPUT_CELLS];
sprintf(filename, "weights2_cell%d.txt", o);

FILE *f = fopen(filename, "w+");

int i;
for(i=0;i<HIDDEN_UNITS;i++)
{
fprintf(f, "%.5g\n",Gl->output_layer.cell[o].weight[i]);
}

fclose(f);
}

printf(" weights2 have been exported \n\n");

///   ###################### export bias in output cells ################################

//writing  to file.

char filename_bias2[]="bias2.txt";

FILE *fbias = fopen(filename_bias2, "w+");
for(i=0;i<NUMBER_OF_OUTPUT_CELLS;i++)
{
fprintf(fbias, "%.5g\n",Gl->output_layer.cell[i].bias);
}
fclose(f);

printf("biases2 have been exported \n\n");


}
