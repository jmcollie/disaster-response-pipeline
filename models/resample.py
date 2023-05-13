"""
A module for resampling a training set.
"""


# Author: Jonathan Collier


import numpy as np
from typing import NamedTuple
from sklearn.utils import resample


class UnderRepresentedClass(NamedTuple):
    """Stores underrepresented classes from the dependent variables of the
    training set.
    """
    index : int
    class_ : int
    class_count: int


class Resampler:
    """Class for resampling from the training set.
    
    Parameters
    ----------
    X : numpy.ndarray
        The independent variables from the training set.
    y : numpy.ndarray
        The dependent variables from the training set.
    
    Attributes
    ----------
    X : numpy.ndarray
        Stores the independent variables from the training set.
    y : numpy.ndarray
        Stores the dependent variables from the training set.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __call__(self):
        self.resample()
        
        return self.X, self.y
    
    def resample(self):
        """
        Determines the unique arrays in the instance attribute `y`
        and samples evenly from them. If any class in `y` will be 
        underrepresented the number of samples taken is changed to
        1000. 
        
        Parameters
        ----------
        None 
        
        Returns
        -------
        None
        """
        row_arrays, counts = np.unique(self.y, return_counts=True, axis=0)
        X_resample = []; y_resample = []
        underrepresented_classes = self.get_underrepresented_classes()
        
        for row_array, count in zip(row_arrays, counts): 
            
            n_samples = int(self.y.shape[0]/len(row_arrays))

            for underrepresented_class in underrepresented_classes:
                if row_array[underrepresented_class.index] == underrepresented_class.class_:
                    n_samples = 1000
                    break
            
            indices = np.where((self.y==row_array).all(axis=1))

            # Sampling without replacement.
            X, y = resample(
                self.X[indices], 
                self.y[indices], 
                n_samples=min(n_samples, count), 
                replace=False
            )
            X_resample.append(X)
            y_resample.append(y)
                
            # Sampling with replacement.
            X, y = resample(
                self.X[indices], 
                self.y[indices], 
                n_samples=max(n_samples-count, 0),
                replace=True
            )
            X_resample.append(X)
            y_resample.append(y)
            

        self.X = np.concatenate(X_resample, axis=0)
        self.y = np.concatenate(y_resample, axis=0)
    
    
    def get_underrepresented_classes(self):
        """
        Identifies classes that are represented in less than 1% 
        (severe imbalance) of unique arrays in attribute `y`, and 
        returns an instance of `UnderRepresentedClass`.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        underrepresented_class : UnderRepresentedClass
            A class that is underrepresented in the unique arrays 
            of instance attribute `y`.
        """
        arrays, counts = np.unique(self.y, return_counts=True, axis=0)
        underrepresented_classes = []
        for index in range(arrays.shape[1]):
            classes, class_counts = np.unique(arrays[:, index], return_counts=True, axis=0)
            for class_, class_count in zip(classes, class_counts):
                if class_count/len(arrays) < .01:
                    underrepresented_classes.append(
                        UnderRepresentedClass(
                            index, 
                            class_, 
                            class_count
                        )
                    )
        return underrepresented_classes
