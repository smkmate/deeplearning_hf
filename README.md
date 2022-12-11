 **Group Name:** \*Rise of AI\* <br>
 **Group Members:**   
- Tugyi Beatrix (T63K63), 
- El-Ali Maya (BHI5LF), 
- Simkó Máté (O3BMRX)

# I. Milestone

## Project description:
### **Name**: Face Recognition and Generation
We will create a model for face feature recognition and for face generation with specific parameters.<br>
In this notebook we created the data for this task.<br>
We load the FairFace dataset and its labels and split them into Test, Valid and Train datasets.
<br>
We created a custom dataset for handling the images, and we performed transformations on them.<br>
At the and we created DataLoaders for the datasets and visualized one batch of the train data.
	

## The repository contains only one file for this task: 
CheckPoint1/FaceRecognition.ipynb - Can be run by clicking the *Open in Colab* link


# II. Milestone

# The repository contains only one file for this Milestone, with the training and evaluation process: 

CheckPoint2/FaceRecignition_base.ipynb - Can be run by clicking the *Open in Colab* link

In this notebook, first we load and preprocess the data and then define the training and evaluating functions.

We can set the hyperparameters in the next section and for training we have to run the fit_one_cycle() function.
After each epoch we can see the validation and training loss.

In the end we can measure the performance of the trained model with the evaluate() function and see the results
in a confusion matrix.

# Final Milestone

For the final submission six models were trained one for each classification task, on each base model. The hyperparameter optimisation, training and evaluation of these models can be seen in Final with the finalmilestone_{base_model}\_{task}.ipynb naming convention. These can be run by clicking *Open in Colab* link.

The trained models are saved in TrainedModels.

The report can be found in documentation.pdf and its source is documentation.docx

The demo app created for the testing of the models can be seen in FaceRecognitionApp for which further instructions can be found in FaceRecognitionApp/README.md.
