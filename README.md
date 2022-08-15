# Disaster-Pipeline-Project

## Project motivation
My motivation for this project was to work on my data engineering, natural language processing and machine learning skills as part of the Udacity data science nanodegree to build a model to for an API that classifies disaster messages.

Appen (previously Figure-8) provided pre labeled tweets and text messages from a real disaster, I visualised the data and classified the data into 36 different categories which could then be used the messages which could be used by disaster responders in an emergency situation. 

## Installations
The project was built using Python and uses packages which should 
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

## Licensing, Authors and Acknowledgements
Credit due to Udacity and Appen. Udacity for the training and the starter code and Figure-8 for the data. If you would like to use any code or data please cite Udacity, Appen and myself.
