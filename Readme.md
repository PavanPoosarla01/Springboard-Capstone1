# Text Analysis and Automatic Rating Prediction Using TED Data
Author : Pavan Poosarla,  Capstone Project 1

Springboard Data Science Career Track, 2019-20

## Description
This project automatically categorizes TED talks into four most popular rating categories, namely, Fascinating, Beautiful, Informative and Funny. It also has an additional category, 'BadTalk' which is a catchall for talks not falling into any of the other categories. Model utilizes the entire transcript of the talk and also additional metadata including the attached tags, duration, type of TED event, etc. The audience reactions are also extracted from the transcript and are added as separate features to the model. This is completed as Capstone Project 1 for the Springboard data Science career track.

## File Organization
All the files and code are organized into folders. See below for info on each folder
* Notebooks : _Includes all the jupyter notebooks used for coding. The suggeted order is DataWrangling, Storytelling, Statistics, Text Analysis, and Modelling_
* data : _raw, intermin and final data as csv files_
* reports : _Final report and presentation. Also includes interim working documents_

## Conclusion
The project has significant pre processing of the data to extract audience reactions, get summary statistics of ratings, audience reactions, tags and ratings. Also has a Naive bayes based methodology to predict the wpords in transcript strongly predictive of each rating. Fianlly, three different supervised learning algorithms are used for prediction : 1. Naive bayes, 2. Random Forest, and 3. Logistic Regression. After optimization, Logistic regression seems to have the best performance from amongst the three models


