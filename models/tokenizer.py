"""
A module for preprocessing and tokenizing text.
"""

# Author: Jonathan Collier


import nltk
import re
import gensim
import gensim.downloader as api
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords
from nltk.metrics.distance import edit_distance

embeddings = api.load('glove-wiki-gigaword-50')
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'universal_tagset'])
stopwords = set(stopwords.words('english'))


class Preprocessor():
    """A class for preprocessing text.
    
    Parameters
    ----------
    stopwords : set
        Stopwords from `nltk.corpus.stopwords`.
    categories : pandas.core.indexes.base.Index
        Categories from dependent variable.
    embeddings
        The pre-trained word vectors from `glove-wiki-gigaword-50`.

    Attributes
    ----------
    stopwords : set
        Stores stopwords from `nltk.corpus.stopwords`.
    categories : pandas.core.indexes.base.Index
        Stores categories from the dependent variable.
    embeddings
        Stores the pre-trained word vectors from `glove-wiki-gigaword-50`.
    """
    def __init__(self, stopwords, categories, embeddings):
        self.stopwords = stopwords
        self.embeddings = embeddings
        
        # Setting topics using input parameter `topics`.
        self.find_related_categories(categories)

    def __call__(self, document):
        
        # Setting text to lowercase.
        document = document.lower()
        
        # Removing special characters from text.
        document = re.sub('\\W', ' ', document)
        
        tokens = []
        for word in word_tokenize(document):
            if word in self.stopwords:
                continue
            else:
                checked_word = self.spell_check(word)
                if checked_word:
                    tokens.append(checked_word)
        return ' '.join(tokens)
    
    def spell_check(self, word):
        """
        Checks whether `word` is valid in attribute `embeddings`, 
        otherwise, checks whether the word with minimum edit_distance 
        between `word` and `categories` is valid using `embeddings`. 
        Returns `word`, the word with the minimum edit distance, or None.
        
        Parameters
        ----------
        word : str
            An individual word from a given document.
        
        Returns
        -------
        : str or None
            The parameter `word`, the word with the minimum edit distance, 
            or None.
        
        """
        if word in self.embeddings:
            return word
        elif word in self.stopwords:
            return None
        else:
            word_ = self.edit_distance(word)
            if word_ in self.embeddings:
                return word_
            else:
                return None
   
    def edit_distance(self, word):
        """
        Computes the edit distance between the parameter `word` 
        and each topic in the attribute `categories`. Returns the first 
        word with the minimum edit distance if the edit distance is 
        less than or equal to 3, otherwise, returns None.
        
        Parameters
        ----------
        word : str
            The word used compute the edit distance against each category
            in attribute `categories`. 
         
        Returns
        -------
        category : str | None
            The first category with the minimum edit distance, or None.
        
        """
        edit_distances = [
            (edit_distance(word, category, substitution_cost=2), category) 
            for category in self.categories if category[0] == word[0]
        ]

        if edit_distances: 
            distance, category = sorted(
                edit_distances, 
                key = lambda value : value[0]
            )[0]
            if distance <= 3:
                return category
        return word
               
    def find_related_categories(self, categories):
        """
        Finds categories related to input parameter `categories`. Assigns each 
        category and any related categories to attribute `categories`.

        Parameters
        ----------
        categories 
            The categories of each dependent variable.
            
        Returns
        -------
        None
            Assigns the parameter `categories` and related categories
            to instance attribute `categories`.
            
        """
        related_categories = set()
        for category in categories:
            category_split = category.split('_')
            for word, _ in self.embeddings.most_similar(positive=category_split, topn=10):
                related_categories.add(word)
                [related_categories.add(word) for word in category_split]
        self.categories = related_categories

    
class Tokenizer():
    """A class for tokenizing and lemmatizing text.
    
    Parameters
    ----------
    category_names : pandas.core.indexes.base.Index
        The category names of the dependent variables.
    
    Attributes
    ----------
    wnl : nltk.WordNetLemmatizer
        An instance of `nltk.stem.WordNetLemmatizer` class.
    preprocessor : Preprocessor
        An instance of the `Preprocessor` class.
    
    """
    def __init__(self, categories):
        self.wnl = WordNetLemmatizer()
        self.preprocessor = Preprocessor(
            stopwords=stopwords, 
            categories=categories,
            embeddings=embeddings
        )
        
        
    def __call__(self, document):
        document = self.preprocessor(document)
        return [
            self.wnl.lemmatize(token[0], self.tag_converter(token[1])) 
            for token in pos_tag(word_tokenize(document), tagset='universal')
        ]
    
    
    @staticmethod
    def tag_converter(tag):
        """Converts the universal tag returned by the `nltk.pos_tag` function into 
        a single character tag used by `nltk.stem.WordNetLemmatizer.lemmatize` method.
        
        Parameters
        ----------
        tag : str
            The universal tag returned from `nltk.pos_tag`.
        Returns
        -------
        : str
            Character tag for passing to `nltk.stem.WordNetLemmatizer().lemmatize` method.
        """
        if tag == 'ADJ': 
            return 'a'
        elif tag == 'ADJ_SAT':
            return 's'
        elif tag == 'VERB':
            return 'v'
        elif tag == 'ADV':
            return 'r'
        else:
            return 'n'