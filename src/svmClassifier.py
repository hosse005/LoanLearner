#!/usr/bin/python3

import sys
sys.path.append( '..' )
from featureExtractor import FeatureExtractor
from learningAgent import LearningAgent
from sklearn import svm
import numpy as np

class SVMClassifier( LearningAgent ):
    ''' 
    Support Vector Machine implementation of the LearningAgent base class
    '''

    def __init__( self , featureExtractor, kernel='rbf' ):
        '''
        @param featureExtractor: FeatureExtractor object for fetching
        preprocessed training data
        @param kernel: Type of kernel to use with the SVM
        '''
        
        # Invoke the super's constructor with the FeatureExtractor
        super().__init__( featureExtractor )

        # Set default regularization parameter to be 1
        self.reg = 1

        # Set the kernel type
        self.kernel = kernel

        # Create the classifier
        '''
        @param C: inverse of regularization, larger C -> lower regularization
        @param kernel: type of kernel to be used in the SVM
        @param probability: enables probability output capability for the 
        classifier, increases time to learn
        '''
        self.clf = svm.SVC( C=self.reg, kernel=self.kernel, probability=True )


    def trainModel( self ):
        '''Train the classifier with the X_train and y_train members'''
        
        # Log status - TODO: move this to a logging class
        print( 'Training on %d samples w/ SVM (%s kernel)' % 
               ( len( self.X_train ), self.kernel ) )
        
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
        self.clf = svm.SVC( C=self.reg, kernel=self.kernel, probability=True )

        
    def setKernelType( self , kernel ):
        '''Setter for kernel type'''
        self.kernel = kernel

        # Re-configure the classifier
        self.clf = svm.SVC( C=self.reg, kernel=self.kernel, probability=True )


    def getClfCoeffs( self ):
        '''Return classifier learning weights'''
        return self.clf.coef_

        
    def __del__( self ):
        pass
