import json
import plotly
import pandas as pd
import numpy as np
import regex as re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine
from nltk.corpus import stopwords, words
STOPWORDS = set(stopwords.words('english'))
VALID_WORDS = set(words.words())

plotly.io.renderers.default = 'browser'

import sys
sys.path.append('../models')
from tokenizer import Tokenizer
from train_classifier import recall_scorer


app = Flask(__name__)


    
def tokenize(document):
    """
    A simple function for tokenizing text.
    
    Parameters
    ----------
    document
    
    Returns
    -------
    tokens 
        The tokenized message.
    
    """
    
    document = document.lower()
    document = re.sub('\\W', ' ', document)
    
    tokens = []
    for sentence in sent_tokenize(document):
        for word, tag in pos_tag(word_tokenize(sentence)):
            if {word}.issubset(STOPWORDS):
                continue
            elif {word}.issubset(VALID_WORDS):
                tokens.append(word)     
    return tokens


def plot_class_distribution(Y, category_names, column_index):
    """
    Creates a pie chart of the class distribution by category.
    
    Parameters
    ----------
    Y : numpy.array
        The dependent variables.
    category_names :  numpy.array  
        The category names of the dependent variables.
    column_index : int
        The column index of Y.

    Returns
    -------
    graph
        A pie chart of the class distribution by category.
    """
    classes, counts = np.unique(Y[:, column_index], return_counts=True)
    
    title = "Class Distribution Of Disaster Response Messages"
       
    subtitle = """<span style=\"color:#696969\">
    Share of messages in each class for the selected category (<i>{}</i>).
    </span>
    """.format(
        str(category_names[column_index]).title()
    )
    
    graph=dict(
        data=[Pie(
            labels=classes,
            values=counts,
            text=['Class {}'.format(class_) for class_ in classes],
            hole=.4,
            marker=dict(
                colors=['rgb(116,195,101)', 'rgb(247,180,107)', 'rgb(215,220,128)']  #d7dc80
            ),
            textfont=dict(
                size=16,
                color='#696969'
            ),
            textinfo="text+percent",
            textposition="outside",
            rotation=90,
            domain=dict(
                x=[0, 1],
                y=[0, .9]
            )
        )],
        layout=dict(
            title=dict(
                text="{}<br><sup>{}</sup>".format(title, subtitle),
                font=dict(
                    size=25
                ),
                x=0.5
            )
        ),
        config = dict(
            responsive=True
        )
    )
    return graph


def plot_top_words_by_category(X, Y, category_names, column_index=0):
    """
    Creates a bar chart of the top words by category.
    
    Parameters
    ----------
    X : numpy.array
        The independent variables.
    Y : numpy.array
        The dependent variables.
    category_names : list
        The category names of the dependent variables.
    column_index : int
        The column index of Y.
    
    Returns
    -------
    graph
        A plotly bar chart of the top words by category.
    """
    classes, counts = np.unique(Y[:, column_index], return_counts=True)
    min_index = np.argmin(counts)
    min_class = np.min(classes[min_index])
    
    # Apply CountVectorizer to get word counts across all documents.
    vectorizer = CountVectorizer(tokenizer=tokenize)
    cv = vectorizer.fit_transform(X[Y[:, column_index] == min_index][:, 0])
    
    # Sum counts by word across documents, sort, and select the top 10 words.
    data = pd.DataFrame(
            cv.toarray(), 
            columns=vectorizer.get_feature_names_out()
    ).sum(axis=0).sort_values(ascending=False).head(10)
    
    
    title = "10 Most Common Words In Disaster Response Messages"
    subtitle = "<span style=\"color:#696969\">" \
    "Frequency of the 10 most common words in the minority class " \
    "(<i>Class {}</i>) for the selected category " \
    "(<i>{}</i>).</span>".format(
        min_class,
        str(category_names[column_index]).title()
    )


    graph = dict(
        data=[Bar(
            x=data.index,
            y=data.values,
            name=category_names[column_index],
            marker=dict(
                color='rgb(116,195,101)' 
            )
        )],
        layout = dict(
            title=dict(
                text='{}<br><sup>{}</sup>'.format(title, subtitle),
                font=dict(
                    size=25
                )
            ),
            yaxis=dict(
                title=dict(
                    text='Word Frequency',
                    font=dict(
                        size=16,
                        color='#696969'
                    ),
                    automargin=True
                ),
                ticklabelposition='outside left',
                tickfont=dict(
                    color='#696969',
                    size=16
                ),
                gridcolor="#bcbcbc",
                griddash="dash",
                gridwidth="1.2"
            ),
            xaxis=dict(
                tickfont=dict(
                    color='#696969',
                    size=16
                ),
                automargin=True
            ),
            autosize=True
        ),
        config = dict(
            responsive=True
        )
        
    )
    return graph


@app.route('/graphs', methods=['GET', 'POST'])           
def generate_plots():
    """
    Returns the graphs and div ids in JSON format.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    graphJSON : json
        The graphs and div ids in JSON format.
    
    """
    column_index = int(request.args['selection'])
    print(column_index)
    X = df[['message', 'genre']].values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns
    
    # Get Graph information.
    class_distribution_plot = plot_class_distribution(Y, category_names, column_index)
    top_words_plot = plot_top_words_by_category(X, Y, category_names, column_index)



    graphJSON = json.dumps(
        dict(
            plots=[class_distribution_plot, top_words_plot],
            ids=['graph-1', 'graph-2']
        ),
        cls=plotly.utils.PlotlyJSONEncoder
    )
    
    return graphJSON
    
    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('pipeline', engine)


# load model
model = joblib.load("../models/classifier.pkl")



@app.route('/')
@app.route('/index', methods=['POST', 'GET'])
def index():
    """
    Renders the html template master.html. 
    
    Parameters
    ----------
    None
    
    Returns
    -------
    Rendered html template master.html.
    """
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns


    # render web page with plotly graphs
    return render_template(
        'master.html', 
        categoryLen=len(category_names),
        categoryNames=list(category_names)
    )


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Renders the html template go.html. 
    
    Parameters
    ----------
    None
    
    Returns
    -------
    Rendered html template go.html.
    """
    # save user input in query
    query = request.args.get('query', '') 
    
    # use model to predict classification for query
    classification_labels = model.predict(np.array([query, 'direct']).reshape(1, -1))[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()