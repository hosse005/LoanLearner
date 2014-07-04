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
        
        # Loop over all test data and assert proper conversion
        for row in self.mFeatureExtractor.getTrainingData():
            int_rate = self.mFeatureExtractor.intRateConversion( row )
            self.assertFalse( re.search( '%', str( int_rate ) ) )

    def test_loanGradeHash( self ):
        '''Test loan grade hashing function'''

        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'sub_grade' )
        
        # Hardcode test dictionary
        mTestDict = {'A1': 1,  'A2': 2,  'A3': 3,  'A4': 4,  'A5': 5,
                     'B1': 6,  'B2': 7,  'B3': 8,  'B4': 9,  'B5': 10,
                     'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15,
                     'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
                     'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25,
                     'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30,
                     'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35 }

        # Loop over all test data and assert correct hash is returned
        for row in self.mFeatureExtractor.getTrainingData():
            sub_grade_hash = self.mFeatureExtractor.loanGradeHash( row )
            testKey = row[idx]
            self.assertEqual( mTestDict[testKey], sub_grade_hash )


if __name__ == '__main__':
    unittest.main()
