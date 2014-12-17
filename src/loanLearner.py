#!/usr/bin/python3

import sys
import argparse
import time
import pickle
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from inputReader import InputReader
from lendingClubFeatureExtractor import LendingClubFeatureExtractor
from logisticClassifier import LogisticClassifier
from svmClassifier import SVMClassifier
from dTreeClassifier import DecisionTreeClassifier

# Application version
''' Revision History
0.0.1 = First working implemenation - logistic regression
0.0.2 = Added first SVM learner
0.0.3 = Added a decision tree classifier
'''
appVersion = '0.0.3'

# Default input source must be relative to this main entry script
defaultInput = '../res/LendingClubFeatureExtractorTest.csv'

# File with input samples to predict outcome
predictInput = '../tmp/predictInputSamples.csv'

# Classifier dump location - MUST BE SAME AS the learningAgent's reference
clfDumpLoc = '../tmp/clf.pickle'

# Scaler dump location - MUST BE SAME AS the learningAgent's reference
scalerDumpLoc = '../tmp/scaler.pickle'

# Application entry and dependency injection
def main():
	
    # Construct an argument parser for cmd line interaction
    parser = argparse.ArgumentParser( description = 'This is a risk valuation \
    software package for loan analysis.  The software is used for predicting  \
    whether a given applicant is likely or not to repay a given loan.' )

    # Application version readback option
    parser.add_argument( '-v', '--version', action='version', 
                         version=appVersion )

    # Option to pass in an input file to be processed
    parser.add_argument( '-i', '--input', dest='inputFile',
                         help='Input File Name', required=False,
                         default=defaultInput )

    # Option to specify the type of learning agent to be used
    parser.add_argument( '--classifier', dest='cls',
                         help="Machine Learning classifier type. \n \
                         Current possible options are: \n \
                         'logistic'(default), 'SVM', 'dTree'", 
                         required=False, default='logistic' )

    # Option to specify the SVM kernel to be used
    parser.add_argument( '-k', '--kernel', dest='kernel',
                         help="SVM kernel type. \n Possible options are: \n \
                         'linear', 'poly', 'rbf'(default), or 'sigmoid' ", 
                         required=False, default='rbf' )

    # Option to specify the test fraction used for learning
    parser.add_argument( '--testFraction', dest='tstFrac',
                         help="Fraction of data to be used for test, must be \
                         between 0 and 1", required=False, default=0.2 )

    # Option to specify pre-training dump file
    parser.add_argument( '-d', '--dump', dest='dumpFile', 
                         help='File location for pre-trained data dump', 
                         required=False )

    # Option to specify learning regularization parameter
    parser.add_argument( '-C', '--reg', dest='reg', 
                         help='Classifier regularization parameter', 
                         required=False , default=1 )

    # Option to specify filter path
    parser.add_argument( '--filter', dest='filterPath', 
                         help='Feature Filter resource file', 
                         required=False , default='../res/FeatureFilter.csv' )

    # Option to predict output of some input sample(s)
    parser.add_argument( '-p', '--predict', dest='predict',
                         help="Run application in prediction mode. \
                         Prediction input samples must be located at \
                         ${PROJ_DIR}/tmp/predictInputSamples.csv \
                         (must first train a classifier!!)", required=False, 
                         action='store_true')
    
    # Grab the inputs passed
    args = parser.parse_args()
    m_inputFile = args.inputFile
    m_cls = args.cls
    m_kernel = args.kernel
    m_tstFrac = float(args.tstFrac)
    m_reg = float(args.reg)
    if args.dumpFile is not None:
        m_dumpFile = args.dumpFile
    else:
        m_dumpFile = None
    m_filter = args.filterPath
    m_predict = args.predict

    # Generate time stamp for performance monitoring
    t0 = time.time()

    # Branch on predict flag
    if m_predict is False:
        # Construct the InputReader w/ our input file
        mInputReader = InputReader( m_inputFile )

        # Next, construct our LendingClubFeatureExtractor object
        mFeatureExtractor = LendingClubFeatureExtractor( mInputReader, 
                                                         m_filter )

        # Use the FeatureExtractor to convert the data for learning
        mFeatureExtractor.extractFeatures()
        mFeatureExtractor.applyFeatureFilter()

        # Dump pre-trained data if specified by user
        if m_dumpFile is not None:
            mFeatureExtractor.setOutCSVPath( m_dumpFile )
            mFeatureExtractor.writeFeaturesToCSV()

        # Construct a LearningAgent based on user input
        if m_cls == 'SVM':
            mLearningAgent = SVMClassifier( mFeatureExtractor, m_kernel )
        elif m_cls == 'logistic':
            mLearningAgent = LogisticClassifier( mFeatureExtractor )
        elif m_cls == 'dTree':
            mLearningAgent = DecisionTreeClassifier( mFeatureExtractor )
        else:
            print( 'Invalid classifier passed.  See --help for valid options' )
            return

        # Set the test fraction of data to use for validation
        mLearningAgent.setTstFraction( m_tstFrac )

        # Set the learning regularization parameter
        mLearningAgent.setRegularization( m_reg )

        # Apply preprocessing to the training samples
        mLearningAgent.shuffleSamples()
        mLearningAgent.sampleSlice()
        mLearningAgent.standardizeSamples()

        # Train the classifier and report the accuracy against the test subset
        mLearningAgent.trainModel()
        print( 'Cross Validation accuracy on the test subset = %0.3f' % 
               mLearningAgent.crossValidate() )

        # Dump the classifier object to file
        mLearningAgent.dumpClassifier()

        # Print out the classifier coefficients
        if m_cls == 'logistic' or m_cls == 'dTree':
            print('Classifier coefficients:')
            print(mLearningAgent.getClfCoeffs())

        # Generate end time stamp and report processing time
        t1 = time.time()
        total = t1 - t0
        print( 'Total processing time = %3.2f seconds' % total )

    # Predict flag set, try read a stored classifier and push our inputs 
    # through it
    else:
        # Construct an input reader
        mInputReader = InputReader( predictInput )

        # Next, construct our LendingClubFeatureExtractor object
        mFeatureExtractor = LendingClubFeatureExtractor( mInputReader, 
                                                         m_filter )

        # Use the FeatureExtractor to convert the data
        mFeatureExtractor.extractFeatures()
        mFeatureExtractor.applyFeatureFilter()

        # Dump pre-trained data if specified by user
        if m_dumpFile is not None:
            mFeatureExtractor.setOutCSVPath( m_dumpFile )
            mFeatureExtractor.writeFeaturesToCSV()

        # Try to read in the stored classifier
        try:
            clf = joblib.load( clfDumpLoc )
        except FileNotFoundError:
            print( 'Error! No classifier binary file found.' )
            print( 'Did you train a classifier yet??' )
            return
            
        # Get the output idx and remove the appropriate column from the input
        outputIdx = mFeatureExtractor.listIdx( 'loan_status' )
        data = mFeatureExtractor.getTrainingData()
        data = np.delete( data, outputIdx, 1 )

        # Standarize data to zero mean and unit variance - this should ideally
        # be same as classifier's scaling factor
        with open( scalerDumpLoc, 'rb' ) as f:
            scaler = pickle.load(f)

        data = scaler.transform( data )

        # Loop through all of the inputs and make a prediction
        print()
        for sample in data:
            if clf.predict( sample ) == 0:
                prediction = 'Charged Off'
                result = 0
            else:
                prediction = 'Fully Paid'
                result = 1

            print( 'Predicted outcome of loan: %s' % prediction )
            print( 'Certainty in outcome is: %.1f percent' % 
                   ( clf.predict_proba(sample)[0][result] * 100 ) )
            print()

        print( clf )
        print()

        
if __name__ == '__main__':
    main()

