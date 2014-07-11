#!/usr/bin/python3

from abc import ABCMeta, abstractmethod
import numpy as np
import random

class LearningAgent( metaclass=ABCMeta ):
    ''' 
    Abstract base class for processing training data and generating predictions.
    Implementation classes must implement trainModel(), crossValidate(), 
    and genPrediction() appropriately for the given machine learning
    subclass.
    '''

    def __init__( self, mFeatureExtractor ):
        '''
        Constructor - arguments passed from main
        @param mFeatureExtractor: FeatureExtractor object for getting 
        preprocessed training data
        '''

        # Get training data from FeatureExtractor
        self.trainingData = np.copy( mFeatureExtractor.getTrainingData() )

        # Get output index from FeatureExtractor
        self.y_idx = mFeatureExtractor.listIdx( 'loan_status' )

        # Set the test fraction to default value
        self.tstFraction = 0.2

        # Separate data into subsets with default parameters
        self.sampleSplice( self.tstFraction )


    def sampleSlice( self, fraction ):
        '''
        Split data into training and test subsets
        @param fraction: 0 to 1 fraction of training data to be used for
        learning validation
        '''

        # First, check that fraction argument is between 0 and 1
        if fraction < 0 or fraction > 1 or fraction is None:
            fraction = 0.2
            print('No or bad value passed to LearningAgent.sampleSplice !')
            print('Default value of 0.2 being assigned to fraction')

        # Get sample length and subset boundary
        nSamples = len( self.trainingData )
        tst_idx = nSamples - int( fraction * nSamples )
        
        # Assign member data based on calculated test index
        self.X_train = self.trainingData[:tst_idx]
        self.X_train = np.delete( self.X_train, [:,self.y_idx] )
        self.y_train = self.trainingData[:tst_idx,self.y_idx]

        self.X_test = self.trainingData[tst_idx:]
        self.X_test = np.delete( self.X_test, [:,self.y_idx] )
        self.y_test = self.trainingData[tst_idx:,self.y_idx]
        

    def shuffleSamples( self ):
        '''Shuffle training sample order'''
        # Use python random module for shuffling data
        self.trainingData = random.shuffle( self.trainingData )

        # Reassign training and test subsets
        self.sampleSlice( self.tstFraction )
    

    def setTstFraction( self, fraction ):
        '''Allow for test subset fraction to be set'''
        self.tstFraction = fraction


    def setTrainingData( self, data ):
        '''Allow for training data to be updated'''
        self.trainingData = data


    @abstractmethod
    def trainModel( self ):
        ''' This method is to be implemented by subclasses'''
        pass


    @abstractmethod
    def crossValidate( self ):
        ''' This method is to be implemented by subclasses'''
        pass


    @abstractmethod
    def genPrediction( self ):
        ''' This method is to be implemented by subclasses'''
        pass


    def __del__( self ):
        '''No Destructor implementation'''
        pass

