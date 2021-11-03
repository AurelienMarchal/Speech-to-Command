# Speech-to-Command
Internship project 2021

READ THE FINAL REPORT FOR ALL THE INFORMATION


# Populate dataset

In the populate_dataset.py file change speaker to your name and the nb_iterations to any number of time you want to record every command.

Then run "python populate_dataset.py". (make sure you install all the required libraries)

The recording process should begin telling you the command to say in 5 seconds (this can be changed). If the program detects more than one voiced section it will cancel the recording and start over.

# Training

In the hparam.yaml make sure all the paths are set correctly, especially the data_folder. 

Run pip install speechbrain if you haven't already. 

Run "python train.py hparams/hparam.yaml" (add --device=cpu at the end if you don't have a good GPU).

or Run "python train_with_wav2vect.py hparams/hparam_wav2vect.yaml"

This should start the training process and create the result folder.

# Training with Colab

If you want to run the training process with Colab copy the Speech-to-Command-Notebook.ipynb in the notebook folder in your Google Drive (don't copy all the repo in your drive) and open it with Google Colab. Make sure that the notebook is using GPU and run the cells.

# Infering (use the model)

You will need a trained model before you make predictions.

Make sure you put the right path to the save folder of your model in the hparam_inference.yaml file.

Run "python infering.py hparams/hparam_inference.yaml" and the program will ask if you want to record. When you are ready press enter and you will have 5 seconds to record your command in one go.
