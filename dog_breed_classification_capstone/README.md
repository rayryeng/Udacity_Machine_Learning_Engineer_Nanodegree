# Dog Breed Classification Capstone Project

## Proposal and Report
The proposal for the project can be found in `Capstone_Project_Proposal.pdf` in PDF form or `Capstone_Project_Proposal.md` in Markdown form.  Similarly, the final report can be found in `Capstone_Project_Final_Report.pdf` in PDF form or `Capstone_Project_Final_Report.md`.

## Installing the necessary modules

If it is your desire to run any of the code found in this directory, you can use the `requirements.txt` file to install the necessary modules prior to running.  In your terminal, please navigate to this directory after you've cloned this repo and do:

`$ pip install -r requirements.txt`

It is recommended you do this in a virtual environment or new conda environment so it doesn't interfere with your base Python install.

## Running the Exploration and Training Notebook
The notebook for exploration and training is found in the `dog_breed_classification_capstone/Dog_Identification_App.ipynb` file.  This notebook contains all of the exploratory analysis, model training and validations made for the final project.   Simply run all of the cells from top to bottom to reproduce the results seen in the final report.  Also take note that all datasets that are required to be downloaded are performed in cells that can be executed in the aforementioned notebook.  Please consult the notebook to find the relevant links to the datasets used for this project.  Expect that if you are using a relatively modern GPU for the notebook to take a few hours to complete.  Please do not attempt to run this in CPU mode.

## Running the demo notebook

If you'd like to skip the above step, a demo notebook demonstrating the final classification model's use on arbitrary images can be found in `dog_breed_classification_capstone/Demo_Dog_Breed_Classification.ipynb`, which relies on the `model_transfer.py` file which defines helper functions and the architecture used for the final version of the project.  Open up the aforementioned notebook file and simply replace the `url` string with one of your own that points to an image of a dog.  The notebook will download it, convert the image to a compatible PyTorch tensor and run inference on it.  It will then report the predicted breed of the image.