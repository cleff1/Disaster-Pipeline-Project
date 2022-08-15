# coding=utf-8
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
import sys
import pickle

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """
    data read from sqlite file paths and into pandas dataframe
    
    Inputs:
    messsages_filepath and categories_filepath
    
    Returns:
    data frame of inputs merged
  """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_category', engine)
    
    X = df['message']
    Y = df.iloc[:, 4:]
    categories = list(df.columns[4:])
    
    return X, Y, categories

    
def tokenize(text):
    
    """
    Tokenizes and lemmitizes the text given
    
    text = the text to be tokenized
    
    Returns
    clean_tokens = tokens that are normalised, have no preceeding or following white space.
    
    """
    #tokenize text
    tokens = word_tokenize(text)
    
    #lemmatize text
    lemmatizer = WordNetLemmatizer()
    
    #save clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    
    """
    Finds best parameter using Gridsearch and makes classifier 
   
    """
    
    #Construct pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier (RandomForestClassifier()))
    ])
    
    #get parameters
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    #Use gridearch method
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
   
    """
    Evaluate model and report on perfromance of model
    
    Inputs -
    Model
    X_test
    Y_test
    Categories
    
    Output -
    classification_report
    """
    
    # build classification report on each column
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.values[:,i], y_pred[:,i]), '_'*50)

def save_model(model, model_filepath):
    """
    Save model to pickle file
    
    """
    
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()