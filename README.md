# Speech-to-Command
Internship project 2021


# Populate dataset

In the populate_dataset.py file change speaker to your name and the nb_iterations to any number of time you want to record every command.

Then run "python populate_dataset.py". (make sure you install all the required libraries)

The recording process should begin telling you the command to say in 5 seconds (this can be changed). If the program detects more than one voiced section it will cancel the recording and start over.

# Training

In the hparam.yaml make sure all the paths are set correctly, especially the data_folder. 

Run pip install speechbrain if you haven't already. 

Run "python train.py hparam.yaml" (add --device=cpu at the end if you don't have a good GPU).

This should start the training process and create the result folder.
