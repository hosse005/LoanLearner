#!/usr/bin/python3

import sys
sys.path.append( '..' )
from inputReader import InputReader
from lendingClubFeatureExtractor import LendingClubFeatureExtractor
from learningAgent import LearningAgent
import numpy as np
import unittest

# Test array
g_testArray = np.array( [[0.34,   -9.7,   12.2, 1.4e-4, -3.4e5, 5.8e4,
                          0.34,   -9.7,   12.2, 1.4e-4, -3.4e5, 5.8e4],
                         [0.685,  -14.3,  29., -1.4e-4, -4.5e4, 4.9e4,
                          0.685,  -14.3,  29., -1.4e-4, -4.5e4, 4.9e4],
                         [0.2343, -13.33, 25.,  1.4e-5, -5.3e5, 1.9e5,
                          0.2343, -13.33, 25.,  1.4e-5, -5.3e5, 1.9e5]] )

# Test resource must be relative to class under test - Not used
testFile = '../../res/LendingClubFeatureExtractorTest.csv'

class DummyLearningAgentImpl( LearningAgent ):
    ''' Dummy Learning Agent class used for unit test of the base class'''

    def __init__( self , featureExtractor ):
        '''@param featureExtractor: FeatureExtractor object'''

        # Invoke the super's constructor with the FeatureExtractor
        super().__init__( featureExtractor )

    def trainModel( self ):
        '''Dummy implementation'''

    def crossValidate( self ):
        '''Dummy implementation'''

    def genPrediction( self ):
        '''Dummy implementation'''
        
    def __del__( self ):
        pass


class LearningAgentTest( unittest.TestCase ):

    def setUp( self ):
        '''Set up the dependencies for the test execution'''

        # Construct an InputReader and FeatureExtractor for dependency injection
        self.mInputReader = InputReader( testFile )
        self.mFeatureExtractor = LendingClubFeatureExtractor( self.mInputReader
                                                            )

        # Push our local test data into the FeatureExtractor
        self.mFeatureExtractor.setTrainingData( g_testArray )

        # Now, construct the class under test with the FeatureExtractor
        self.mLearningAgent = DummyLearningAgentImpl( self.mFeatureExtractor )


    def test_getTrainingData( self ):
        '''Test getTrainingData() function returns correct data'''
        np.testing.assert_array_equal( self.mLearningAgent.getTrainingData(),
                                       g_testArray )
                                           
if __name__ == '__main__':
    unittest.main()
