#!/usr/bin/python3

import sys
import argparse
from inputReader import InputReader
from lendingClubFeatureExtractor import LendingClubFeatureExtractor
from logisticClassifier import LogisticClassifier

# Application version
appVersion = '0.0.1'

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
                         help='Input File Name', required=False )

    # TODO - add option to specify pre-training dump file
    '''
    parser.add_argument( '-d', '--preTrainDump', dest='dumpFile',
                         help='Pretrained data dump', required=False )
    '''

    # Grab the inputs passed
    args = parser.parse_args()

    # Process inputs
    if args.inputFile != None:
        m_inputFile = args.inputFile
    else:
        m_inputFile = defaultInput

    # Construct the InputReader w/ our input file
    mInputReader = InputReader( m_inputFile )

    # Read the data from the input file
    mInputReader.readFile()

    # Next, construct our LendingClubFeatureExtractor object
    mFeatureExtractor = LendingClubFeatureExtractor( mInputReader )

    # Use the FeatureExtractor to convert the data for learning
    mFeatureExtractor.extractFeatures()

    # Construct a LogisticClassifier for learning
    mLearningAgent = LogisticClassifier( mFeatureExtractor )

    # Apply preprocessing to the training samples
    mLearningAgent.shuffleSamples()
    mLearningAgent.sampleSlice()
    mLearningAgent.standardizeSamples()

    # Train the classifier and report the accuracy against the test subset
    mLearningAgent.trainModel()
    print( 'Cross Validation accuracy on the test subset = %0.2f' % 
           mLearningAgent.crossValidate() )


if __name__ == '__main__':
    main()

