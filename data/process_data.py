import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Imports the `messages` and `categories` csv files into
    DataFrames, performs an inner join, and returns the combined
    DataFrame.
    
    Parameters
    ----------
    messages_filepath : str
        The filepath of the messages.csv file.
    categories_filepath : str
        The filepath of the categories.csv file.
    Returns
    -------
    : pandas.DataFrame
        A DataFrame of the messages and categories
        datasets joined together.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how='inner', on='id')


def clean_data(df):
    """
    Creates a separate column for each unique category in the 
    `categories` column in `df` with a One-Hot Encoded value 
    for whether the category is present in the row, drops the
    `category` column, and removes duplicate rows.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of the messages and categories datasets merged.
    
    Returns
    -------
    df : pandas.DataFrame
        Input parameter `df` concatenated with a DataFrame of categories
        parsed out of the `categories` column in input parameter `df`,
        with the `categories` column dropped, and duplicate rows removed.
    """
    # Creating a DataFrame from the values in the categories
    # column of df.
    categories = df['categories'].str.split(';', expand=True)
    
    # Parsing and setting column names of the DataFrame using the first row.
    first_row = categories.iloc[0]
    categories.columns = first_row.apply(lambda value: 
                                         value.split('-')[0]).values

    for column in categories:
        # Setting each value to be the last character of the string.
        categories[column] = categories[column].str.slice(start=-1)
        
        # Converting column datatype from string to numeric.
        categories[column] = pd.to_numeric(categories[column])
        
    # Dropping the categories column from df.
    df.drop(columns=['categories'], inplace=True)
    
    # Concatenating df and categories.
    df = pd.concat([df, categories], axis=1)
    
    # Dropping duplicated rows.
    df.drop(index=df[df.duplicated()].index, inplace=True)
        
    return df


def save_data(df, database_filepath):
    """
    Creates a sqlite database using `database_filepath` and loads
    DataFrame `df` to the created database with a
    table name of `pipeline`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame of messages and categories.
    database_filepath : 
        The name of the database to save the data to.

    Returns
    -------
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('pipeline', engine, index=False, if_exists='replace')

def main():
    """
    Calls `load_data`, `clean_data` and `save_data` methods,
    if sys.argv does not equal four, an error message is 
    printed.
    
    Parameters
    ----------
    None

    Returns
    -------
    None

    """
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