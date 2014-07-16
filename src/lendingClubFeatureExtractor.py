#!/usr/bin/python3

import sys
sys.path.append( '..' )
from inputReader import InputReader
from featureExtractor import FeatureExtractor
import numpy as np
import csv
import re
from datetime import datetime

class LendingClubFeatureExtractor( FeatureExtractor ):
    ''' 
    LendingClub implementation of the FeatureExtractor base class
    '''

    def __init__( self , inputReader ):
        '''@param inputReader: InputReader object for fetching raw data'''

        # Invoke the super's constructor with the InputReader
        super().__init__( inputReader )

        # Set the feature set which needs conversion - TODO: resource driven
        self.featureConvLookup = {'term', 'int_rate', 'sub_grade', 'emp_length',
                                  'home_ownership', 'is_inc_v', 'loan_status',
                                  'purpose', 'addr_state', 'bc_util', 
                                  'earliest_cr_line', 'revol_util'}


    def termConversion( self, training_sample ):
        '''Enumerate loan term duration'''
        
        # Get index of loan term feature
        idx = self.listIdx( 'term' )
        
        # Check expression and convert appropriately
        if re.search( '36', training_sample[idx] ):
            return 36
        else:
            return 60


    def pcntRemove( self, training_sample, feature ):
        '''Remove '%' from raw data'''
        
        # Get index of passed feature
        idx = self.listIdx( feature )

        return float( re.sub( '%', '', training_sample[idx] ) )


    def loanGradeHash( self, training_sample ):
        '''Hash A1-G5 subgrade ratings to 1 - 35'''
        
        # Get index of loan subgrade feature
        idx = self.listIdx( 'sub_grade' )

        # Search for all possible letter and number grades and modify tmp
        tmp = 0
        mLetterDict = {'A': 0, 'B': 5, 'C': 10, 'D': 15, 
                       'E': 20, 'F': 25, 'G': 30}
        
        # Search by letter grade first
        match = re.search( '[ABCDEFG]', training_sample[idx] )
        if match:
            key = match.group()
            tmp = mLetterDict[key]
        else:
            raise ValueError( 'Unexpected value read from sub_grade @ training \
            sample %d' % idx )
            return

        # Add number subgrade to base letter grade dict value
        match = re.search( '[12345]', training_sample[idx] )
        if match:
            tmp += int( match.group() )
        else:
            raise ValueError( 'Unexpected value read from sub_grade @ training \
            sample %d' % idx )
            return

        return tmp


    def empLengthConversion( self, training_sample ):
        ''' Convert employment length to suitable integer value'''

        # Get index of employment length feature
        idx = self.listIdx( 'emp_length' )

        # Search for number of years
        match = re.findall( '[<\+n123456789]', training_sample[idx] )
        
        if match:
            # Take the last match for '10+' differentiation from '1'
            tmp = match[-1]

            # Assign 0.1 to '< 1 year' and 20 to '10+ years' to accentuate
            if tmp == '<':
                return 0.1
            elif tmp == '+':
                return 20
            elif tmp == 'n':
                return 0
            else:
                return int( tmp )
        else:
            raise ValueError( 'Unexpected value read from emp_length @ training\
            sample %d' % idx )
            return

        
    def homeOwnershipEnumerator( self, training_sample ):
        '''Enumerate home ownership statuses'''

        # Get index of home ownership feature
        idx = self.listIdx( 'home_ownership' )

        # Search for expected values
        match = re.search( 'RENT|OWN|MORTGAGE|OTHER', training_sample[idx] )

        if match:
            tmp = match.group()

            # Assign integers to the possible statuses
            if tmp == 'RENT':
                return 1
            elif tmp == 'MORTGAGE':
                return 2
            elif tmp == 'OWN':
                return 3
            elif tmp == 'OTHER':
                return 4
            else:
                return 0
        else:
            raise ValueError( 'Unexpected value read from home_ownership @ \
            training sample %d' % idx )
            return

        
    def incomeVerifiedConversion( self, training_sample ):
        '''Convert income verification status to binary value'''

        # Get index of income verification feature
        idx = self.listIdx( 'is_inc_v' )

        # Search for 'not', indicating source not verified
        match = re.search( 'Not', training_sample[idx] )

        if match:
            return 0
        else:
            return 1


    def purposeEnumerator( self, training_sample ):
        '''Enumerate loan purpose features'''

        # Get index of income verification feature
        idx = self.listIdx( 'purpose' )

        # Create an enumeration dictionary, try to enum from most to least
        # credible and leave slot for others in the middle
        purposeDict = {'car': 9, 'credit_card': 10, 'debt_consolidation': 5, 
                       'education': 4, 'home_improvement': 2, 'house': 1,
                       'major_purchase': 8, 'medical': 3,
                       'small_business': 7, 'vacation': 12, 'wedding': 11}
        
        # Search for expected values
        regex = '|'.join( ['car', 'credit_card', 'debt_consolidation',
                           'education', 'home_improvement', 'house', 
                           'major_purchase', 'medical', 'small_business',
                           'vacation', 'wedding'] )

        match = re.search( regex, training_sample[idx] )

        if match:
            purpose = match.group()
            return purposeDict[purpose]
        else:
            return int(len(purposeDict) / 2) + 1


    def stateEnumerator( self, training_sample ):
        '''Enumerate state feature'''

        # Get index of income verification feature
        idx = self.listIdx( 'addr_state' )

        # Create an enum dictionary - TODO: need to determine importance of
        # assigned value to learning algorithm performance
        stateDict = {'AK': 1,  'AL': 2,  'AR': 3,  'AZ': 4,  'CA': 5,  
                     'CO': 6,  'CT': 7,  'DC': 8,  'DE': 9,  'FL': 10, 
                     'GA': 11, 'HI': 12, 'IA': 13, 'ID': 14, 'IL': 15,
                     'IN': 16, 'KS': 17, 'KY': 18, 'LA': 19, 'MA': 20,
                     'MD': 21, 'ME': 22, 'MI': 23, 'MN': 24, 'MO': 25,
                     'MS': 26, 'MT': 27, 'NC': 28, 'ND': 29, 'NE': 30,
                     'NH': 31, 'NJ': 32, 'NM': 33, 'NV': 34, 'NY': 35,
                     'OH': 36, 'OK': 37, 'OR': 38, 'PA': 39, 'PR': 40,
                     'RI': 41, 'SC': 42, 'SD': 43, 'TN': 44, 'TX': 45,
                     'UT': 46, 'VA': 47, 'VI': 48, 'VT': 49, 'WA': 50,
                     'WI': 51, 'WV': 52, 'WY': 53}

        return stateDict[training_sample[idx]]
        

    def earlyCrLineConversion( self, training_sample ):
        '''Earliest line of credit conversion, w/ respect to system epoch'''

        # Get index of earliest credit line feature
        idx = self.listIdx( 'earliest_cr_line' )

        # Convert the date to a datetime object        
        earlyCrLine = datetime.strptime( training_sample[idx],
                                         "%m/%d/%Y  %H:%M" )

        # Convert datetime object to float seconds since epoch
        earlyCrLine -= datetime(1970, 1, 1)

        return earlyCrLine.total_seconds()


    def statusConversion( self, training_sample ):
        '''
        Assign 'Charged Off' to 0, and 'Fully Paid' to 1 for classification
        Remove any other features from the training set which don't have
        one of these two statuses; Client code needs to check for not
        defined status before use
        @return status: 0 = charged off, 1 = fully paid, 2 = not defined
        '''

        # Get index of loan status feature
        idx = self.listIdx( 'loan_status' )

        # Search for desired status values - TODO: possibly add late statuses to
        # negative classification as well
        match = re.search( 'Charged Off|Fully Paid', training_sample[idx] )

        # If we have a match, return conversion, otherwise remove the sample
        if match:
            match = match.group()
            if match == 'Charged Off':
                return 0
            else:
                return 1
        else:
            return 2


    def extractFeatures( self ):
        '''Convert training data to format suitable for learning where needed'''
        
        # Create a dirt set for removing samples
        mDirtSet = set()

        # Loop through all training samples and run conversions - TODO: this
        # could/should have a parallel implementation for performance
        for i, training_sample in enumerate( self.trainingData ):
            
            # First digitize output and remove unclassified samples
            idx = self.listIdx( 'loan_status' )
            loanStatus = self.statusConversion( training_sample )

            if loanStatus == 0 or loanStatus == 1:
                training_sample[idx] = loanStatus
            else:
                # Mark dirty sample
                mDirtSet.add( i )
                continue

            # If we have a valid sample, run through all conversions and update
            # trainingData

            # Loan term conversion
            idx = self.listIdx( 'term' )
            try:
                training_sample[idx] = self.termConversion( training_sample )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )
            
            # Loan interest rate conversion
            idx = self.listIdx( 'int_rate' )
            try:
                training_sample[idx] = self.pcntRemove( training_sample, 
                                                        'int_rate' )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )

            # Revolving utility conversion
            idx = self.listIdx( 'revol_util' )
            try:
                training_sample[idx] = self.pcntRemove( training_sample, 
                                                        'revol_util' )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )

            # Loan grade hash
            idx = self.listIdx( 'sub_grade' )
            try:
                training_sample[idx] = self.loanGradeHash( training_sample )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )

            # Employment length conversion
            idx = self.listIdx( 'emp_length' )
            try:
                training_sample[idx] = self.empLengthConversion(
                    training_sample )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )
            
            # Home ownership enumeration
            idx = self.listIdx( 'home_ownership' )
            try:
                training_sample[idx] = self.homeOwnershipEnumerator( 
                    training_sample )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )

            # Income verification conversion
            idx = self.listIdx( 'is_inc_v' )
            try:
                training_sample[idx] = self.incomeVerifiedConversion(
                    training_sample )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )

            # Loan purpose enumeration
            idx = self.listIdx( 'purpose' )
            try:
                training_sample[idx] = self.purposeEnumerator( training_sample )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )

            # State enumeration
            idx = self.listIdx( 'addr_state' )
            try:
                training_sample[idx] = self.stateEnumerator( training_sample )
            except ( ValueError, KeyError ):
                # Mark dirty sample
                mDirtSet.add( i )

            # Earliest credit line conversion
            idx = self.listIdx( 'earliest_cr_line' )
            try:
                training_sample[idx] = self.earlyCrLineConversion( 
                    training_sample )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )

            # Finally, convert all training data to float type
            # Remove the sample if it throws an exception
            try:
                training_sample = training_sample.astype( float )
            except ValueError:
                # Mark dirty sample
                mDirtSet.add( i )

        # Remove all marked dirty samples
        self.nRmvSamples = len( mDirtSet )
        self.trainingData = np.delete( self.trainingData, list( mDirtSet ), 0 )
        print(mDirtSet)
        print( 'Number of samples removed = %d' % self.nRmvSamples )
        self.trainingData = self.trainingData.astype( float )

    def __del__( self ):
        pass


