import sys
import pandas as pd
import nltk
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, make_scorer, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tokenizer import Tokenizer
from resample import Resampler


def load_data(database_filepath):
    """
    Loads data from the database_filepath and
    returns the `X`, `Y`, and `category_names` variables.
    
    Parameters
    ----------
    database_filepath
        The filepath of the database to load data from.
    
    Returns
    -------
    X : numpy.ndarray
        The indepedent variables.
    Y : numpy.ndarray
        The dependent variables.
    category_names : pandas.core.indexes.base.Index
        The category names of the dependent variables.
    
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    data = pd.read_sql('pipeline', con=engine)
    X = data[['message', 'genre']].values
    category_names = data.drop(columns=['id', 'message', 'original', 'genre']).columns
    Y = data.drop(columns=['id', 'message', 'original', 'genre']).values
    
    
    return X, Y, category_names


def recall_scorer(Y_test, Y_pred):
    """
    Calculates the averaged unweighted recall.
    
    Parameters
    ----------
    Y_test : numpy.ndarray
        The Y test set.
    
    Y_pred : numpy.ndarray
        The predicted Y values.
        
    Returns
    -------
    : float
        The averaged unweighted recall.
    
    """
    scores = []
    for index in range(Y_test.shape[1]):
        score = recall_score(Y_test[:, index], Y_pred[:, index], average=None, zero_division=1)
        scores.append(np.mean(score))
    return np.mean(scores)


def build_model(category_names):
    """
    Builds the model using `sklearn.model_selection.GridSearchCV`
    for hyperparameter tuning.
    
    Parameters
    ----------
    category_names : pandas.core.indexes.base.Index
        The category names of the dependent variables.
    Returns
    -------
    : sklearn.model_selection.GridSearchCV
        The GridSearchCV model with hyperparameters for testing.
    """

    # Creates the CountVectorizer and TfidfTransformer pipeline.
    tfidf_pipeline = Pipeline([
        ('vect', CountVectorizer(
                    tokenizer=Tokenizer(
                        categories=category_names
                    ), 
                    min_df=2)
                ), 
        ('tfidf', TfidfTransformer())  
    ])

    # Pipeline for one-hot encoding genre.
    category_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='first'))
    ])

    # ColumnTransformer combining both preprocessing steps.
    tfidf_preprocessing = ColumnTransformer([
        ('tfidf_pipeline', tfidf_pipeline, 0),
        ('cat_pipeline', category_pipeline, [1])
    ])

    pipeline = Pipeline([
        ('preprocessing', tfidf_preprocessing),
        ('clf', None)
    ])

    parameters = [
        {
            'preprocessing__tfidf_pipeline__vect__min_df': (1, 2, 3),
            'preprocessing__tfidf_pipeline__vect__max_df': (0.8, 0.9, 1.0),
            'preprocessing__tfidf_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
            'clf': [MultiOutputClassifier(AdaBoostClassifier())],
            'clf__estimator__learning_rate': [.01, .1, 1]
        }
    ]
                 
    scorer = make_scorer(recall_scorer)
                         
    return GridSearchCV(
        pipeline, 
        param_grid=parameters, 
        scoring=scorer, 
        n_jobs=-1, 
        cv=3,
        verbose=2
    ) 
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Uses the model to calculate `Y_pred` using `X_test`. Outputs
    the classification report comparing `Y_pred` and `Y_test`
    for each category in `category_names`.
    
    Parameters
    ----------
    model
        The classificiation model found using the `build_model` function.
    X_test : numpy.ndarray
        The X test set.
    Y_test : numpy.ndarray
        The Y test set.
    category_names : pandas.core.indexes.base.Index
        The categories that correspond to each output in `Y_test`.
    
    Returns
    -------
    None 
        Prints the classification_report for each column.
    """
    Y_pred = model.predict(X_test)
    
    for index in range(Y_pred.shape[1]):
        print(category_names[index])
        print(classification_report(Y_test[:, index], Y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Saves the trained classification model to a pickle file.
    
    Parameters
    ----------
    model 
        The classificiation model found using the `build_model` function.
    model_filepath
        The filepath for saving the pickled model.

    Returns
    -------
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Resampling the training set...')
        rs = Resampler(X=X_train, y=Y_train)
        X_train, Y_train = rs()
        
        print('Building model...')
        model = build_model(category_names)
        
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