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

    def test_empLengthConversion( self ):
        '''Test employment length function'''

        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'emp_length' )
        
        # Loop over all test data and assert correct conversion is returned
        for row in self.mFeatureExtractor.getTrainingData():
            emp_length = self.mFeatureExtractor.empLengthConversion( row )
            
            # Convert function calculated text back to expected string
            if emp_length == 0.1:
                emp_length = '<'
            elif emp_length == 20:
                emp_length = '10'
            elif emp_length == 0:
                emp_length = 'n/a'
            else:
                emp_length = str( emp_length )
            
            # Use converted emp_length for reg exp test against test resource
            self.assertTrue( re.search( emp_length, row[idx] ) )

    def test_homeOwnershipEnumerator( self ):
        '''Test home ownership enumeration'''

        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'home_ownership' )

        # Test dictionary
        mTestDict = {1: 'RENT', 2: 'MORTGAGE', 3: 'OWN', 4: 'OTHER'}
        
        # Loop over all test data and assert correct enumeration is returned
        for row in self.mFeatureExtractor.getTrainingData():
            homeOwnE = self.mFeatureExtractor.homeOwnershipEnumerator( row )

            # Convert function result back to expected string
            if homeOwnE == 1 or 2 or 3 or 4:
                homeOwn = mTestDict[homeOwnE]
            else:
                homeOwn = 'FAIL'
            
            # Use converted homeOwn for reg exp test against test resource
            self.assertTrue( re.search( homeOwn, row[idx] ) )

    def test_incomeVerifiedConversion( self ):
        '''Test income verification conversion'''

        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'is_inc_v' )

        # Loop over all test data and assert correct conversion is returned
        for row in self.mFeatureExtractor.getTrainingData():
            is_inc_v = self.mFeatureExtractor.incomeVerifiedConversion( row )
         
            # Assert that when is_inc_v is 0, input contains 'Not' string
            if is_inc_v == 0:
                self.assertTrue( re.search( 'Not', row[idx] ) )
            else:
                self.assertFalse( re.search( 'Not', row[idx] ) )
        


if __name__ == '__main__':
    unittest.main()
