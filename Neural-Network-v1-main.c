 /**
 * @file main.c
 * @mainpage Neural Network in Embedded System.
 * @brief Testing the Computation of Neural Network in Embedded System. [SM32F030R8T6] chip.
 * @author Waleed Ahmed Daud.
 * @Website waleed-daud.github.io
 * @Linkedin https://linkedin/in/waleed-daud-78472b9b
 * @Email waleed.daud@outlook.com
 * @date OCT 21
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "screen.h"
#include "mnist-utils.h"
#include "mnist-stats.h"
#include "NeuralNetwork.h"



/****************************************************************************************************************************/

/**
 * @details

 */

void Test_Neural_Network(GeneralLayer *Gl){
// open MNIST files
        FILE *imageFile, *labelFile;
        imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
        labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);


        // screen output for monitoring progress
        displayImageFrame(7,5);

        int errCount = 0;
        double cost=0;
        // Loop through all images in the file
        int imgCount;
    for ( imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){

        // display progress
        displayLoadingProgressTesting(imgCount,5,5);

        // Reading next image and corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);

        // set target Output of the number displayed in the current image (=label) to 1, all others to 0
        Vector targetOutput;
        targetOutput = getTargetOutput(lbl);

        displayImage(&img, 8,6);


        int predictedNum =Prediction(Gl,&img);

        if (predictedNum!=lbl) errCount++;

        printf("\n      Prediction: %d   Actual: %d ",predictedNum, lbl);

        /// #############################   Compute Cost    ############################
        cost+=Cost_Function(Gl,&targetOutput);

        displayProgress(imgCount, errCount, 5, 66);

    }

         cost=cost/MNIST_MAX_TESTING_IMAGES;

         FILE *f;
         f = fopen("Testing_report.txt", "a");

         fprintf(f,"################################# Total Report ###########################################  \n");
         printf(f,"\n  Accuracy: %.7g \n",(MNIST_MAX_TESTING_IMAGES-errCount/MNIST_MAX_TESTING_IMAGES)*100);
         fprintf(f,"Result: Correct=%5d  Incorrect=%5d  \n",imgCount+1-errCount, errCount);
         fprintf(f, "Cost: %.7g\n",cost);

    // Close files
    fclose(f);
    fclose(imageFile);
    fclose(labelFile);


}


/**
 * @details Tests a layer by looping through and testing its cells
 * Exactly the same as TrainLayer() but WITHOUT LEARNING.
 * @param l A pointer to the layer that is to be training
 */

double Neural_Network(GeneralLayer *Gl,MNIST_Image *img, Vector *targetOutput){
   /// ########################  Forward Propagation     #########################
        Forward_Propagation(Gl,img);
  /// #############################   Compute Cost    ############################
        double c=Cost_Function(Gl,targetOutput);
  /// ######################### Backward Propagation #############################
        Backward_Propagation(Gl,targetOutput);
  ///  ######################## Update Parameters ################################
        Update_Weights(Gl);

        return c;
    }





    ///###################################################################################################



/**
 * @details Main function to run MNIST-1LNN
 */

int main(int argc, const char * argv[]) {

    // remember the time in order to calculate processing time at the end
    time_t startTime = time(NULL);
    // clear screen of terminal window
    clearScreen();
    printf("#################################### Beginning ##########################################");


    /// #######################################  (General Layer) ##############################################
        GeneralLayer general_layer;
    /// #######################################   Parameters initialization             #######################
        initLayer(&general_layer);


    /// #######################################       Training          #########################################

      int iteration;

      for(iteration=0;iteration<NUMITERATIONS;iteration++)
      {
        printf("########################################### Iteration %d ##############################################",iteration);
        double cost=0;

        Vector targetOutput; /// a Label will convert to 0-1 format eg: 5 -> 0000100000.
        int errCount = 0; /// error counter.
       /// ###################################### Image Processing ########################################
        /// open MNIST files
        FILE *imageFile, *labelFile;
        imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
        labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);

        /// screen output for monitoring progress
        displayImageFrame(5,5);

        /// Loop through all images in the file.
        int imgCount;
        for ( imgCount=0; imgCount<MNIST_MAX_TRAINING_IMAGES; imgCount++)
        {

        /// display progress
           displayLoadingProgressTraining(imgCount,3,5);

        /// Reading next image and corresponding label
            MNIST_Image img = getImage(imageFile);
            MNIST_Label lbl = getLabel(labelFile);

        /// set target Output of the number displayed in the current image's label to 1, all others to 0

            targetOutput = getTargetOutput(lbl);

            displayImage(&img, 6,6);

        /// ############################# Neural Network ###################################################################
            cost+=Neural_Network(&general_layer,&img,&targetOutput);


            int predictedNum = getPrediction(&general_layer);
            if (predictedNum!=lbl) errCount++;

            printf("\n      Prediction: %d   Actual: %d \n",predictedNum, lbl);

            displayProgress(imgCount, errCount, 3, 66);

        }



        /// ###########################################  Cost Function  #######################################################


             printf("############################# Cost training image #################################### \n\n\n");

             cost=(cost/MNIST_MAX_TRAINING_IMAGES);
             FILE *f;
             f = fopen("Training_report.txt", "a");

             fprintf(f,"\n ################################# Iteration number:%5d ###########################################  \n\n",iteration);
             fprintf(f,"################################# Total Report ###########################################  \n");
             fprintf(f,"Result: Correct=%5d  Incorrect=%5d  \n",imgCount+1-errCount, errCount);
             fprintf(f, "Cost: %.7g\n",cost);

             fclose(f);
             printf("Cost in iteration %d: %lf \n\n",iteration,cost);
             delay(2000);

            fclose(imageFile);
            fclose(labelFile);



    }

    /// #################################################  Testing  #################################################

        Test_Neural_Network(&general_layer);

        locateCursor(38, 5);
        export_Weights(&general_layer);

        /// Calculate and print the program's total execution time
        time_t endTime = time(NULL);
        double executionTime = difftime(endTime, startTime);
        printf("\n    DONE! Total execution time: %.1f sec\n\n",executionTime);

        ///  ###################### free the allocated space used in init_layer function  ###################


        return 0;
}

