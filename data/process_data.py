import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Merge and load csv files into a dataframe
    
    Input:
    message_filepath
    categories_filepath
    
    Output:
    dataframe of merged files
    
    """
    #Load message file
    messages = pd.read_csv(messages_filepath)
    
    #Load categories file
    categories = pd.read_csv(categories_filepath)
    
    #Merge datasets
    df = pd.merge(messages, categories, on = 'id')
    
    return df


def clean_data(df):
    """
    Clean data by splitting values in the categories column on ; to create separate columns
    
    Inputs: df of merged datasets
    
    Outputs: cleaned df
    
    """
    #Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True )
    
    #Select the first row of the categories dataframe and extract list of new column names
    row = row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    
    #Rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just 0 or 1
    for column in categories:
       
        categories[column] = categories[column].astype('str').str.replace('2', '1')
        categories[column] = categories[column].astype('int')
        
    #Drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    #Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)
    
    #Convert categories to binary
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    
    #Drop duplicates
    df = df.drop_duplicates()
   
    return df


def save_data(df, database_filename):
    """
    Save cleaned dataset into sqlite databse
    
    Input: Cleaned and merged dataset
    
    Output: None
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_category', engine, index=False, if_exists = 'replace')
    
   
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
