#!/usr/bin/python3

import sys
sys.path.append( '..' )
from inputReader import InputReader
from featureExtractor import FeatureExtractor
import numpy as np
import csv
import unittest

# Test resource must be relative to class under test
testFile = '../../res/InputReaderTest.csv'
dumpFile = '../../tmp/featureExtractorTest.csv'

class DummyFeatureExtractorImpl( FeatureExtractor ):
    ''' Dummy Feature Extractor class used for unit test of the base class'''

    def __init__( self , inputReader ):
        '''@param inputReader: InputReader object for fetching raw data'''

        # Invoke the super's constructor with the InputReader
        super().__init__( inputReader )

    def extractFeatures( self ):
        ''' Populate members w/ test data'''
        self.rawData = [['Feature', 'Extractor', 'Test'],
                        ['34.', '79.2', '0.3342'],
                        ['0.34', '32', '994.3']]
        self.features = self.rawData[0]
        self.trainingData = np.array( self.rawData[1:] )

    def __del__( self ):
        pass

class FeatureExtractorTest( unittest.TestCase ):

    def setUp( self ):
        '''Set up the dependencies for the test execution'''

        # Construct our InputReader object, pass it the test csv file
        self.mInputReader = InputReader( testFile )

        # Construct the class under test with the InputReader
        self.mFeatureExtractor = DummyFeatureExtractorImpl( self.mInputReader )

        # Extract test features
        self.mFeatureExtractor.extractFeatures()

    def test_getFeatures( self ):
        '''Test getFeatures() function returns correct data'''
        self.assertEqual( self.mFeatureExtractor.getFeatures(), 
                          self.mFeatureExtractor.rawData[0]
                        )

    def test_getTrainingData( self ):
        '''Test getTrainingData() function returns correct data'''

        # Assert that returned data is of type numpy.ndarray
        self.assertTrue( isinstance( self.mFeatureExtractor.getTrainingData(),
                         np.ndarray ) )

        # Assert that returned data value and size are correct
        np.testing.assert_array_equal( self.mFeatureExtractor.getTrainingData(),
                                       np.array(
                                        self.mFeatureExtractor.rawData[1:] ) )
    
    def test_dumpToCSV( self ):
        '''Test debug dump and path set'''
        
        # Set up test path and write test data
        self.mFeatureExtractor.setOutCSVPath( dumpFile )
        self.mFeatureExtractor.writeFeaturesToCSV()
        
        # Read out test data and assert its equivalent to class attribute
        with open( dumpFile ) as csvfile:
            reader = csv.reader( csvfile, delimiter=',' )
            dumpRead = list()
            for row in reader:
                dumpRead.append( row )
            self.assertTrue( dumpRead, self.mFeatureExtractor.rawData )
            csvfile.close()

    def test_listIdx( self ):
        '''Test list index returns correct feature index'''
        for i,v in enumerate( self.mFeatureExtractor.features ):
            self.assertEqual( i, self.mFeatureExtractor.listIdx( v ) )

if __name__ == '__main__':
    unittest.main()

