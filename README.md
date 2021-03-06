# Udacity Machine Learning Engineer Nanodegree Course

This repo contains my work for the Machine Learning Engineer Nanodegree Course on Udacity.  Each course will be partitioned into separate directories.  The exception for this is the first course where it was considered optional.  Specifically, it was to release a Python package to PyPI.  I used the same methods provided in the course and released my work on vanishing point detection to PyPI.  Though it is not available in this repo, you can visit this work here: https://github.com/rayryeng/XiaohuLuVPDetection.

Please note that the code provided is designed to run on Amazon SageMaker.  However, at a minimum you will need Python 3.6, PyTorch 0.4.0 and MXNet.

## SageMaker Deployment Project - Sentiment Analysis

The goal of this project was to work with the IMDB movie review dataset and create a predictive model that can predict whether a movie review is positive or negative.  The instructions for completion and deployment can be found in `sentiment_analysis_model`.  The notebook file found in this directory is the overall assignment, with directions to modify the necessary files to complete the task.

## Plagiarism Detection via SageMaker

The goal of this project was to work with sample data to detect if plagiarism occurred between sample definitions of computer science terms with what Wikipedia stores.  Specifically, custom features such as containment and longest common subsequence were used as input to a machine learning algorithm to help accurate predict whether someone has plagiarised a body of text.  There was a choice between using traditional machine learning algorithms or neural networks through PyTorch.  I decided to use PyTorch.  The relevant preparatory and deployment notebook files, as well as the necessary Python files to facilitate deployment on Amazon SageMaker can be found in `plagiarism_detection`, with directions to modify the necessary files to complete the project.

## Dog Breed Classification Capstone Project

The goal of this project was to create a deep learning classification model that would determine the dog breed of an image of a dog.  If the image is of a human, the system will determine what the closest dog breed is to that human.  We explore a "from scratch" architecture as well as employing transfer learning with a deeper network, as well as perform exploratory data analysis, analyse some statistics of the data and draw conclusions about the model and to justify its readiness for being deployed.

The project can be found in the `dog_breed_classification_capstone` directory.  The proposal for the project as well as the final report can be found here as well.  Finally, the notebook for exploration and training is found in the `dog_breed_classification_capstone/Dog_Identification_App.ipynb` file and a demo notebook demonstrating its use on arbitrary images can be found in `dog_breed_classification_capstone/Demo_Dog_Breed_Classification.ipynb`, which relies on the `model_transfer.py` file which defines helper functions and the architecture used for the final version of the project.