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

    def extractFeatures( self ):
        '''Convert training data to format suitable for learning where needed'''

    def __del__( self ):
        pass
