# Disaster-Pipeline-Project

## Project motivation
My motivation for this project was to work on my data engineering, natural language processing and machine learning skills as part of the Udacity data science nanodegree to build a model to for an API that classifies disaster messages.

Appen (previously Figure-8) provided pre labeled tweets and text messages from a real disaster, I classified the data into 36 different categories and visualised the data, which could then be used be used by disaster responders in an emergency situation. The steps used can be followed here:

1. ETL Pipeline
Write a data cleaning pipeline that loads datasets, merges datasets, cleans the data and stores it in a SQLite database.

2. ML Pipeline
Write a machine learning pipeline that loads the data from SQLlite, splits the dataset into training and test, builds a text processing and ML pipeline, trains and turnes models, outputs results on the test set and exports the final model as a picklefile.

3.Flask app
The flask app is provided by Udacity and you need to modify the file paths for your data and add visualizations. 

## Installations
The project was built using Python and uses packages which are included as part of the Anaconda distribution and Python version 3.6 and higher.

Packages such as nltk, flask, sqlalchemy, sklearn, plotly will need to be installed too and can be found within the python script.  


## File descriptions
    Disaster-Response-Pipeline
      |-- app
            |-- templates
                    |-- go.html # classification result page of web app
                    |-- master.html # main page of web app
            |-- run.py # Flask file that runs app
      |-- data
            |-- disaster_categories.csv # data to process
            |-- disaster_message.csv # data to process 
            |-- DisasterResponse.db # database to save clean data to
            |-- process_data.py
      |-- models
            |-- classifier.pkl # saved model
            |-- train_classifier.py
      |-- README

## Instructions
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

## Visualisations
Here we can see the collected and cleaned data as graphs. You can also see the app which is used to classify messages in to one of the 36 categories in the first image. 
![image](https://user-images.githubusercontent.com/107194172/184644907-31009d29-1228-423f-b13d-d4c09d396bb4.png)
![image](https://user-images.githubusercontent.com/107194172/184644953-896f6a3d-d9b6-446c-af92-17201f6a3f46.png)
![image](https://user-images.githubusercontent.com/107194172/184644983-1c3e928b-8fe4-4329-a8d5-8ededebe9c63.png)




## Licensing, Authors and Acknowledgements
Credit due to Udacity and Appen. Udacity for the training and the starter code and Figure-8 for the data. If you would like to use any code or data please cite Udacity, Appen and myself.
