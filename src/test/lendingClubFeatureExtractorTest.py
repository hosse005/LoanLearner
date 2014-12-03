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
filterTestFile = '../../res/FeatureFilter.csv'

class LendingClubFeatureExtractorTest( unittest.TestCase ):

    def setUp( self ):
        '''Set up the dependencies for the test execution'''

        # Construct our InputReader object, pass it the test csv file
        self.mInputReader = InputReader( testFile )

        # Construct the class under test with the InputReader
        self.mFeatureExtractor = LendingClubFeatureExtractor( 
            self.mInputReader, filterTestFile )


    def test_termConversion( self ):
        '''Test termEnumerator functionality'''
        
        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'term' )
        
        # Loop over all test data and assert proper enumeration
        for row in self.mFeatureExtractor.getTrainingData():
            term = self.mFeatureExtractor.termConversion( row )
            if re.search( '36 months', row[idx] ):
                self.assertEqual( 36, term )
            elif re.search( '60 months', row[idx] ):
                self.assertEqual( 60, term )
            else:
                raise ValueError( 'Encountered unsupported term value' )


    def test_pcntRemove( self ):
        '''Test '%' removal'''
        
        # Loop over all test data and assert proper conversion
        for row in self.mFeatureExtractor.getTrainingData():
            int_rate = self.mFeatureExtractor.pcntRemove( row, 'int_rate' )
            revol_util = self.mFeatureExtractor.pcntRemove( row, 'revol_util' )

            # Assert that no '%' contained in the results
            self.assertFalse( re.search( '%', str( int_rate ) ) )
            self.assertFalse( re.search( '%', str( revol_util ) ) )


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

        
    def test_purposeEnumerator( self ):
        '''Test loan purpose enumeration'''

        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'purpose' )

        # Test dictionary - This must align with UUT dict!!
        mTestDict = {1: 'house', 2: 'home_improvement', 3: 'medical',
                     4: 'education', 5: 'debt_consolidation',
                     7: 'small_business', 8: 'major_purchase', 9: 'car', 
                     10: 'credit_card', 11: 'wedding', 12: 'vacation'}

        # Loop over all test data and assert correct conversion is returned
        for row in self.mFeatureExtractor.getTrainingData():
            purpose = self.mFeatureExtractor.purposeEnumerator( row )

            # Assert string is found in test data based on returned enum
            if purpose in mTestDict.keys():
                self.assertEqual( mTestDict[purpose],
                                  re.search( mTestDict[purpose], row[idx] )
                                  .group() )


    def test_stateEnumerator( self ):
        '''Test state enumeration'''

        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'addr_state' )

        # Test dictionary - This must align with UUT dict!!
        mTestDict = {1:  'AK', 2:  'AL', 3:  'AR', 4:  'AZ', 5:  'CA',  
                     6:  'CO', 7:  'CT', 8:  'DC', 9:  'DE', 10: 'FL', 
                     11: 'GA', 12: 'HI', 13: 'IA', 14: 'ID', 15: 'IL',
                     16: 'IN', 17: 'KS', 18: 'KY', 19: 'LA', 20: 'MA',
                     21: 'MD', 22: 'ME', 23: 'MI', 24: 'MN', 25: 'MO',
                     26: 'MS', 27: 'MT', 28: 'NC', 29: 'ND', 30: 'NE',
                     31: 'NH', 32: 'NJ', 33: 'NM', 34: 'NV', 35: 'NY',
                     36: 'OH', 37: 'OK', 38: 'OR', 39: 'PA', 40: 'PR',
                     41: 'RI', 42: 'SC', 43: 'SD', 44: 'TN', 45: 'TX',
                     46: 'UT', 47: 'VA', 48: 'VI', 49: 'VT', 50: 'WA',
                     51: 'WI', 52: 'WV', 53: 'WY'}

        # Loop over all test data and assert correct conversion is returned
        for row in self.mFeatureExtractor.getTrainingData():
            addr_stateE = self.mFeatureExtractor.stateEnumerator( row )

            # Assert returned value matches w/ test dictionary
            self.assertTrue( re.search( mTestDict[addr_stateE], row[idx] ) )


    def test_earlyCrLineConversion( self ):
        '''Test date conversion'''

        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'earliest_cr_line' )

        # Generate a test date
        testDate = ['01/01/1972  01:50']

        # Time elapsed since 2014
        delta = 42

        # Push null entries into testDate to simulate feature placement in 
        # the training set
        for i in range(idx):
            testDate.insert(0,'')
        
        # Assert time elapsed since epoch is correct for given test time
        self.assertEqual( delta, self.mFeatureExtractor.
                          earlyCrLineConversion( testDate ) )

    def test_statusConversion( self ):
        '''Loan status conversion test'''

        # Grab appropriate column index
        idx = self.mFeatureExtractor.listIdx( 'loan_status' )

        # Loop over all test data and assert correct conversion is returned
        for row in self.mFeatureExtractor.getTrainingData():
            loan_status = self.mFeatureExtractor.statusConversion( row )

            # Assert that correct loan_status is returned based on test input
            if loan_status == 0:
                self.assertTrue( re.search( 'Charged Off', row[idx] ) )
            elif loan_status == 1:
                self.assertTrue( re.search( 'Fully Paid', row[idx] ) )
            else:
                self.assertFalse( re.search( 'Fully Paid|Charged Off', 
                                             row[idx] ) )

    def test_extractFeatures( self ):
        '''Feature extraction test'''

        # Get initial training sample count
        nSamples = self.mFeatureExtractor.getSampleCnt()

        # Invoke feature extraction on our test object
        self.mFeatureExtractor.extractFeatures()
        
        # Make local sample removal count
        nRmvSamples = nSamples - self.mFeatureExtractor.getSampleCnt()

        # Assert local removal calculation corresponds with actual
        self.assertEqual( nRmvSamples, 
                          self.mFeatureExtractor.getRmvSampleCnt() )


if __name__ == '__main__':
    unittest.main()
