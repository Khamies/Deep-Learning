/**
 * @file Neural-Network.c
 * @mainpage Implementation of Neural Network using Genann Library.
 * It's apart of Neural-Network-in-Embedded-Systems Project (it's also in github :) ).
 * @brief Implementation of feed forward Neural Network algorithm using Genann Library and pima-indians-diabetes dataset.
 * @author Waleed Ahmed Daud.
 * @Website waleed-daud.github.io
 * @Linkedin https://linkedin/in/waleed-daud-78472b9b
 * @Email waleed.daud@outlook.com
 * @date OCT 22
 */

#include <stdio.h>
#include <time.h>

#include "genann.h"

#define NUM_OF_TRAINING_OBSERVATIONS 600
#define NUM_OF_TESTING_OBSERVATIONS   168
#define NUM_OF_FEATURES  8
#define NUM_OF_ITERATIONS 1000
#define LEARNING_RATE 0.001
#define NUM_OF_HIDDEN_LAYERS 1
#define NUM_OF_HIDDEN_UNITS  3
#define NUM_OF_OUTPUT_UNITS  1


void delay(unsigned int mseconds)
{
    clock_t goal = mseconds + clock();
    while (goal > clock());
}



int main(int argc, char *argv[])
{
    printf("ANN is working now ... .... ...... ..... ...... ..... \n");
   // printf("Train a small ANN to the XOR function using backpropagation.\n");


    double  train_data[NUM_OF_TRAINING_OBSERVATIONS][NUM_OF_FEATURES];
    double  train_label[NUM_OF_TRAINING_OBSERVATIONS];             /// true train values.
    double test_data[NUM_OF_TESTING_OBSERVATIONS][NUM_OF_FEATURES];
    double  test_label[NUM_OF_TESTING_OBSERVATIONS];             /// true test values.
    double predicted_Train_label [NUM_OF_TRAINING_OBSERVATIONS];
    double predicted_Test_label  [NUM_OF_TESTING_OBSERVATIONS];        /// predicted_Test_label  .


    int i,j;

/// ############################################### Preprocessing #########################################################

/// ################################################### Load train_data #######################################################
int row,column=0;

FILE *fp;
fp=fopen("pima-indians-diabetes.txt","r");

if(fp!=NULL)
{
for(row=0;row<NUM_OF_TRAINING_OBSERVATIONS;row++)
{
    fscanf(fp,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",&train_data[row][column],&train_data[row][column+1],&train_data[row][column+2],&train_data[row][column+3],&train_data[row][column+4],&train_data[row][column+5],&train_data[row][column+6],
           &train_data[row][column+7],&train_label[row]);
}

fclose(fp);

}
else
{
printf("File Can not be opened !");

}

/// ################################################## Load Test Data #######################################################
fp=fopen("pima-indians-diabetes_test.txt","r");
column=0;
 if(fp!=NULL)
{
    for(row=0;row<NUM_OF_TESTING_OBSERVATIONS;row++)
{
            fscanf(fp,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",&test_data[row][column],&test_data[row][column+1],&test_data[row][column+2],&test_data[row][column+3],&test_data[row][column+4],&test_data[row][column+5],&test_data[row][column+6],
                   &test_data[row][column+7],&test_label[row]);
}
fclose(fp);
}

else
{
printf("File Can not be opened !");

}

/*
int j;
for(i=0;i<NUM_OF_TESTING_OBSERVATIONS;i++)
{
    for(j=0;j<NUM_OF_FEATURES;j++)

    {
        printf("%lf \n",test_data[i][j]);
           delay(1000);
    }
    //printf("\n #########################################################");
}
*/

/// ################################################### Network ##############################################################
    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    genann *ann = genann_init(NUM_OF_FEATURES, NUM_OF_HIDDEN_LAYERS, NUM_OF_HIDDEN_UNITS, NUM_OF_OUTPUT_UNITS);

    /* Train on the four train_labeled train_data points many times. */
    for (i = 0; i < NUM_OF_ITERATIONS; ++i)
        {

        for(j=0;j<NUM_OF_TRAINING_OBSERVATIONS;j++)
        {
            genann_train(ann, train_data[j], train_label + j, LEARNING_RATE);
        }

        }

    /// ####################################### * Run the network and see what it predicts. */ ##########################;

    for(i=0;i<NUM_OF_TESTING_OBSERVATIONS;i++)
    {
    predicted_Test_label  [i]=*genann_run(ann, test_data[i]);
     printf("Output for test observation no: [%d] is [%1.f].\n", i, predicted_Test_label  [i]);

    }

    /// ################################################## Training Accuracy #########################################################
    double counter=0;
    for(i=0;i<NUM_OF_TRAINING_OBSERVATIONS;i++)
    {
    if(predicted_Train_label  [i]>0.5) predicted_Train_label  [i]=1;
    else if(predicted_Train_label  [i]<0.5) predicted_Train_label  [i]=0;

    if(predicted_Train_label  [i]==train_label[i])counter++;
    printf("actual: [%lf] ,predicted: [%lf].\n",train_label[i],predicted_Train_label[i]);
    }

    double training_accuracy=(counter/NUM_OF_TRAINING_OBSERVATIONS)*100 ;

    printf("\n\n\n Train Accuracy is: %lf ",training_accuracy);
    delay(2000);
    /// ################################################## Testing Accuracy #########################################################
    counter=0;
    for(i=0;i<NUM_OF_TESTING_OBSERVATIONS;i++)
    {
    if(predicted_Test_label  [i]>0.5) predicted_Test_label  [i]=1;
    else if(predicted_Test_label  [i]<0.5) predicted_Test_label  [i]=0;

    if(predicted_Test_label  [i]==test_label[i])counter++;
    printf("actual: [%lf] ,predicted: [%lf].\n",test_label[i],predicted_Test_label  [i]);
    }

    double accuracy=(counter/NUM_OF_TESTING_OBSERVATIONS)*100 ;

    printf("\n\n\n  Test Accuracy is: %lf ",accuracy);

/// ################################################ Export Weights ##############################################


fp=fopen("Weights.txt","w");

    genann_write(ann,fp);
fclose(fp);
/*
FILE *fp2;
fp2=fopen("Weights.txt","r");
genann *ann2=genann_read(fp2);
if (!ann2) {
        printf("Error loading ANN from file: %s.", "Weights.txt");
        exit(1);
}

printf("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO");
double predicted_Test_label  2[NUM_OF_TESTING_OBSERVATIONS];

    /// ####################################### * Run the network and see what it predicts.  ##########################;

    for(i=0;i<NUM_OF_TESTING_OBSERVATIONS;i++)
    {
    predicted_Test_label  2[i]=*genann_run(ann2, test_data[i]);
     printf("Output for test observation no: [%d] is [%1.f].\n", i, predicted_Test_label  2[i]);

    }

    /// ################################################## Accuracy #########################################################
   counter=0;
    for(i=0;i<NUM_OF_TESTING_OBSERVATIONS;i++)
    {
    if(predicted_Test_label  2[i]>0.5) predicted_Test_label  2[i]=1;
    else if(predicted_Test_label  2[i]<0.5) predicted_Test_label  2[i]=0;

    if(predicted_Test_label  2[i]==test_label[i])counter++;
    printf("actual: [%lf] ,predicted: [%lf].\n",test_label[i],predicted_Test_label  2[i]);
    }

     accuracy=(counter/NUM_OF_TESTING_OBSERVATIONS)*100 ;

    printf("\n\n\n Accuracy2 is: %lf ",accuracy);
    printf("\n\n\n counter2: %lf",counter);

/// ################################################ Export Weights ##############################################


    fclose(fp2);
*/
    genann_free(ann);
    return 0;
}
 
