#!/usr/bin/python3

import sys
sys.path.append( '..' )
from featureExtractor import FeatureExtractor
from learningAgent import LearningAgent
from sklearn import linear_model
from sklearn.externals import joblib
import numpy as np

class LogisticClassifier( LearningAgent ):
    ''' 
    Logistic Regression implementation of the LearningAgent base class
    '''

    def __init__( self , featureExtractor ):
        '''
        @param featureExtractor: FeatureExtractor object for fetching
        preprocessed training data
        '''
        
        # Invoke the super's constructor with the FeatureExtractor
        super().__init__( featureExtractor )

        # Set default regularization parameter to be 1e5
        self.reg = 1e5

        # Create the classifier
        self.clf = linear_model.LogisticRegression( C=self.reg )


    def trainModel( self ):
        '''Train the classifier with the X_train and y_train members'''
        
        # Log status - TODO: move this to a logging class
        print( 'Training on %d samples w/ Logistic Regression' 
               % len( self.X_train ) )
        
        self.clf.fit( self.X_train, self.y_train )


    def crossValidate( self ):
        '''Return the model's accuracy on the test data set'''

        # Log status - TODO: move this to a logging class
        print( 'Testing on %d samples' % len( self.X_test ) )
        
        return self.clf.score( self.X_test, self.y_test ) 
    

    def genPrediction( self , data ):
        '''
        Generate a prediction for any new samples
        @return classification: boolean classification '0' = loan charged off,
                                                       '1' = loan paid
        '''
        assert( isinstance( data, np.ndarray ) )
        return self.clf.predict( data )


    def genProbPrediction( self , data ):
        '''
        Generate the classification probablility for any new samples
        @return cls_list: 0-1 probability of sample belonging to each class
        '''
        assert( isinstance( data, np.ndarray ) )
        return self.clf.predict_proba( data )


    def setRegularization( self, reg ):
        '''Setter for regularization parameter'''
        self.reg = reg
        
        # Re-configure the classifier
        self.clf = linear_model.LogisticRegression( C=self.reg )


    def getClfCoeffs( self ):
        '''Return classifier learning weights'''
        return self.clf.coef_


    def dumpClassifier ( self ):
        ''' Method to serialize and dump the classifier class '''
        joblib.dump( self.clf, self.clfPath )

    def __del__( self ):
        pass

