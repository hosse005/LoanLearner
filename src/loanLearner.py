#!/usr/bin/python3

import sys
import argparse
import time
from inputReader import InputReader
from lendingClubFeatureExtractor import LendingClubFeatureExtractor
from logisticClassifier import LogisticClassifier
from svmClassifier import SVMClassifier

# Application version
''' Revision History
0.0.1 = First working implemenation - logistic regression
0.0.2 = Added first SVM learner
'''
appVersion = '0.0.2'

# Default input source must be relative to this main entry script
defaultInput = '../res/LendingClubFeatureExtractorTest.csv'

# Application entry and dependency injection
def main():
	
    # Construct an argument parser for cmd line interaction
    parser = argparse.ArgumentParser( description = 'This is a risk valuation \
    software package for loan analysis.  The software is used for predicting  \
    whether a given applicant is likely or not to repay a given loan.' )

    # Add version option
    parser.add_argument( '--version', action='version', version=appVersion )


    # Add option to pass in an input file to be processed
    parser.add_argument( '-i', '--input', dest='inputFile',
                         help='Input File Name', required=False,
                         default=defaultInput )

    # Add option to specify the type of learning agent to be used
    parser.add_argument( '--classifier', dest='cls',
                         help="Machine Learning classifier type \n \
                         Current possible options are: \n \
                         'logistic'(default), 'SVM' ", 
                         required=False, default='logistic' )

    # Add option to specify the SVM kernel to be used
    parser.add_argument( '-k', '--kernel', dest='kernel',
                         help="SVM kernel type \n Possible options are: \n \
                         'linear', 'poly', 'rbf'(default), or 'sigmoid' ", 
                         required=False, default='rbf' )

    # TODO - add option to specify pre-training dump file
    '''
    parser.add_argument( '-d', '--preTrainDump', dest='dumpFile',
                         help='Pretrained data dump', required=False )
    '''

    # Grab the inputs passed
    args = parser.parse_args()
    m_inputFile = args.inputFile
    m_cls = args.cls
    m_kernel = args.kernel

    # Generate time stamp for performance monitoring
    t0 = time.time()

    # Construct the InputReader w/ our input file
    mInputReader = InputReader( m_inputFile )

    # Next, construct our LendingClubFeatureExtractor object
    mFeatureExtractor = LendingClubFeatureExtractor( mInputReader )

    # Use the FeatureExtractor to convert the data for learning
    mFeatureExtractor.extractFeatures()

    # Construct a LearningAgent based on user input
    if m_cls == 'SVM':
        mLearningAgent = SVMClassifier( mFeatureExtractor, m_kernel )
    elif m_cls == 'logistic':
        mLearningAgent = LogisticClassifier( mFeatureExtractor )
    else:
        print( 'Invalid classifier passed.  See --help for valid options' )
        return

    # Apply preprocessing to the training samples
    mLearningAgent.shuffleSamples()
    mLearningAgent.sampleSlice()
    mLearningAgent.standardizeSamples()

    # Train the classifier and report the accuracy against the test subset
    mLearningAgent.trainModel()
    print( 'Cross Validation accuracy on the test subset = %0.3f' % 
           mLearningAgent.crossValidate() )

    # Generate end time stamp and report processing time
    t1 = time.time()
    total = t1 - t0
    print( 'Total processing time = %3.2f seconds' % total )

if __name__ == '__main__':
    main()

