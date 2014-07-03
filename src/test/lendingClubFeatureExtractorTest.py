#!/usr/bin/python3

import sys
sys.path.append( '..' )
from inputReader import InputReader
from lendingClubFeatureExtractor import LendingClubFeatureExtractor
import numpy as np
import csv
import re
import unittest

# Test resource must be relative to class under test
testFile = '../../res/LendingClubFeatureExtractorTest.csv'

class LendingClubFeatureExtractorTest( unittest.TestCase ):

    def setUp( self ):
        '''Set up the dependencies for the test execution'''

        # Construct our InputReader object, pass it the test csv file
        self.mInputReader = InputReader( testFile )

        # Construct the class under test with the InputReader
        self.mFeatureExtractor = LendingClubFeatureExtractor( 
            self.mInputReader )

    def test_termEnumerator( self ):
        '''Test termEnumerator functionality'''
        
        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'term' )
        
        # Loop over all test data and assert proper enumeration
        for row in self.mFeatureExtractor.getTrainingData():
            termE = self.mFeatureExtractor.termEnumerator( row )
            if re.search( '36 months', row[idx] ):
                self.assertEqual( 36, termE )
            elif re.search( '60 months', row[idx] ):
                self.assertEqual( 60, termE )
            else:
                raise ValueError( 'Encountered unsupported term value' )

    def test_intRateConversion( self ):
        '''Test interest rate '%' removal'''
        
        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'int_rate' )

        # Loop over all test data and assert proper conversion
        for row in self.mFeatureExtractor.getTrainingData():
            int_rate = self.mFeatureExtractor.intRateConversion( row )
            self.assertFalse( re.search( '%', str( int_rate ) ) )


if __name__ == '__main__':
    unittest.main()
