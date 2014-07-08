#!/usr/bin/python3

import sys
sys.path.append( '..' )
from inputReader import InputReader
from featureExtractor import FeatureExtractor
import numpy as np
import csv
import re

class LendingClubFeatureExtractor( FeatureExtractor ):
    ''' Dummy Feature Extractor class used for unit test of the base class'''

    def __init__( self , inputReader ):
        '''@param inputReader: InputReader object for fetching raw data'''

        # Invoke the super's constructor with the InputReader
        super().__init__( inputReader )

        # Set the feature set - TODO: should prob be resource driven
        # NOTE - this most likely to be replaced w/ features attribute; resource
        #        must only contain columns of interest
        self.featureSet = {'loan_amnt', 'funded_amnt', 'term', 'int_rate',
                           'installment', 'sub_grade', 'emp_length', 
                           'home_ownership', 'annual_inc', 'is_inc_v',
                           'loan_status', 'purpose', 'addr_state',
                           'acc_now_delinq', 'acc_open_past_24mnths',
                           'percent_bc_gt_75', 'bc_util', 'dti', 'delinq_2yrs',
                           'delinq_amnt', 'earliest_cr_line', 'inq_last_6mths',
                           'mths_since_last_delinq',
                           'mths_since_recent_revol_delinq',
                           'mths_since_recent_bc', 'mort_acc', 'open_acc',
                           'pub_rec', 'total_bal_ex_mort', 'revol_bal',
                           'revol_bal', 'revol_util', 'total_bc_limit',
                           'total_acc', 'out_prncp', 'num_rev_accts',
                           'mths_since_recent_bc_dlq', 'num_rec_bankruptcies',
                           'num_accts_ever_120_pd', 'chargeoff_within_12_mths',
                           'tax_liens', 'mths_since_last_major_derog',
                           'num_sats', 'num_tl_op_past_12m', 'mo_sin_rcnt_tl',
                           'tot_hi_cred_lim', 'tot_cur_bal', 'avg_cur_bal',
                           'num_bc_tl', 'num_actv_bc_tl', 'num_bc_sats',
                           'pct_tl_nvr_dlq', 'num_tl_90g_dpd_24m',
                           'num_tl_30dpd', 'num_tl_120dpd_2m'}

        # Set the feature set which needs conversion - TODO: resource driven
        self.featureConvLookup = {'term', 'int_rate', 'sub_grade', 'emp_length',
                                  'home_ownership', 'is_inc_v', 'loan_status',
                                  'purpose', 'addr_state', 'bc_util', 
                                  'earliest_cr_line', 'revol_util'}

    def termEnumerator( self, training_sample ):
        '''Enumerate loan term duration'''
        
        # Get index of loan term feature
        idx = self.listIdx( 'term' )
        
        # Check expression and convert appropriately
        if re.search( '36', training_sample[idx] ):
            return 36
        else:
            return 60

    def intRateConversion( self, training_sample ):
        '''Remove '%' from raw data'''
        
        # Get index of interest rate feature
        idx = self.listIdx( 'int_rate' )

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

        # Add number subgrade to base letter grade dict value
        match = re.search( '[12345]', training_sample[idx] )
        if match:
            tmp += int( match.group() )
        else:
            raise ValueError( 'Unexpected value read from sub_grade @ training \
            sample %d' % idx )

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
        
        
    def extractFeatures( self ):
        '''Convert training data to format suitable for learning where needed'''

    def __del__( self ):
        pass
