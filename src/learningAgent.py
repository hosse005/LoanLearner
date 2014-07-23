#!/usr/bin/python3

from abc import ABCMeta, abstractmethod
from sklearn import preprocessing
import numpy as np

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


    def sampleSlice( self, fraction=None ):
        '''
        Split data into training and test subsets
        @param fraction: 0 to 1 fraction of training data to be used for
        learning validation
        '''

        # First, check that fraction argument is between 0 and 1
        if fraction is None or fraction < 0 or fraction > 1:
            fraction = self.tstFraction

        # Get sample length and subset boundary
        nSamples = len( self.trainingData )
        tst_idx = nSamples - int( fraction * nSamples )
        
        # Assign member data based on calculated test index
        self.X_train = self.trainingData[:tst_idx]
        self.X_train = np.delete( self.X_train, self.y_idx, 1 )
        self.y_train = self.trainingData[:tst_idx,self.y_idx]

        self.X_test = self.trainingData[tst_idx:]
        self.X_test = np.delete( self.X_test, self.y_idx, 1 )
        self.y_test = self.trainingData[tst_idx:,self.y_idx]


    def standardizeSamples( self ):
        '''Standardize training samples to zero mean and unit deviation'''

        # Create a scaler preprocessing object and pass it our training subset
        # Note: scale data w/ training subset and apply to test subset as well
        self.scaler = preprocessing.StandardScaler().fit( self.X_train )
        self.X_train = self.scaler.transform( self.X_train )
        self.X_test = self.scaler.transform( self.X_test )
        

    def shuffleSamples( self , seed=None ):
        '''
        Shuffle training sample order
        @param seed: random number gen repeatability, intended for test
        '''
        
        # First check if were passed a seed
        if seed is not None:
            np.random.seed( seed )

        # Use numpy random module for generating random indices
        indices = np.random.permutation( len( self.trainingData) )

        # Shuffle data based on randomly generated indices
        self.trainingData = self.trainingData[indices]

        # Reassign training and test subsets
        self.sampleSlice( self.tstFraction )
    

    def setTstFraction( self, fraction ):
        '''Allow for test subset fraction to be set'''
        assert( fraction > 0 and fraction < 1 )
        self.tstFraction = fraction


    def setTrainingData( self, data ):
        '''Allow for training data to be updated'''
        assert( isinstance( data, np.ndarray ) )
        self.trainingData = data


    def getTrainingData( self ):
        '''Mechanism for retrieving training data'''
        return self.trainingData


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

    
    @abstractmethod
    def getClfCoeffs( self ):
        ''' This method is to be implemented by subclasses'''
        pass


    def __del__( self ):
        '''No Destructor implementation'''
        pass

