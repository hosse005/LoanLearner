#!/usr/bin/python3

import sys
sys.path.append( '..' )
from inputReader import InputReader
from lendingClubFeatureExtractor import LendingClubFeatureExtractor
from learningAgent import LearningAgent
from math import ceil, fabs
import numpy as np
import unittest

# Test array
g_testArray = np.array( [[0.34,   -9.7,   12.2, 1.4e-4, -3.4e5, 5.8e4,
                          0.34,   -9.7,   12.2, 1.4e-4, -3.4e5, 5.8e4],
                         [0.685,  -14.3,  29., -1.4e-4, -4.5e4, 4.9e4,
                          0.685,  -14.3,  29., -1.4e-4, -4.5e4, 4.9e4],
                         [0.2343, -13.33, 25.,  1.4e-5, -5.3e5, 1.9e5,
                          0.2343, -13.33, 25.,  1.4e-5, -5.3e5, 1.9e5], 
                         [0.792,  -12.2,  33.3, 1.4e-6, -5.2e5, 2.9e4,
                          0.792,  -12.2,  33.3, 1.4e-6, -5.2e5, 2.9e4],
                         [0.792,  -12.2,  33.3, 1.4e-6, -5.2e5, 2.9e4,
                          0.792,  -12.2,  33.3, 1.4e-6, -5.2e5, 2.9e4]] )

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


    def test_sampleSlice( self ):
        '''Test sampleSlice() function correctly splits test and train data'''
        
        # Configure the data to be split evenly for the test
        self.mLearningAgent.sampleSlice( 0.5 )

        # Get the target feature index
        m_yIdx = self.mFeatureExtractor.listIdx( 'loan_status' )

        # Slice boundary
        mBnd = ceil( len(g_testArray) / 2 )
        
        # Generate X_train test array checksum
        m_X_train_sum = np.sum( g_testArray[:mBnd] )
        m_X_train_sum = m_X_train_sum - np.sum( g_testArray[:mBnd, m_yIdx] )

        # Verify sums match w/in some small tolerance
        self.assertTrue( fabs( m_X_train_sum - 
                               np.sum( self.mLearningAgent.X_train ) < 0.001 ) )

        # Generate X_test test array checksum
        m_X_test_sum = np.sum( g_testArray[mBnd:] )
        m_X_test_sum = m_X_test_sum - np.sum( g_testArray[mBnd:, m_yIdx] )
        
        # Verify sums match w/in some small tolerance
        self.assertTrue( fabs( m_X_test_sum - 
                               np.sum( self.mLearningAgent.X_test ) < 0.001 ) )

        # Generate y_train array checksum
        m_y_train_sum = np.sum( g_testArray[:mBnd, m_yIdx] )

        # Verify sums match w/in some small tolerance
        self.assertTrue( fabs( m_y_train_sum - 
                               np.sum( self.mLearningAgent.y_train ) < 0.001 ) )

        # Generate y_test array checksum
        m_y_test_sum = np.sum( g_testArray[mBnd:, m_yIdx] )

        # Verify sums match w/in some small tolerance
        self.assertTrue( fabs( m_y_test_sum - 
                               np.sum( self.mLearningAgent.y_test ) < 0.001 ) )

                                           
if __name__ == '__main__':
    unittest.main()
