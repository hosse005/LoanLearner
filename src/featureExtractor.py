#!/usr/bin/python3

from abc import ABCMeta, abstractmethod
import numpy as np
import csv


class FeatureExtractor( metaclass=ABCMeta ):
    ''' 
    Abstract base class for extracting and generating features from input 
    resource file.  Implementation classes must implement getFeatures() 
    and getTrainingData() appropriately for the given input source.
    '''

    def __init__( self, mInputReader ):
        '''
        Constructor - arguments passed from main
        @param mInputReader: InputReader object for setting raw data
        '''
        # Attribute initialization
        self.rawData = list()
        self.features = list()

        # Feature dump path
        self.outCSVPath = '../../tmp/featureDump.csv'
        
        # Get raw data from the passed InputReader
        self.rawData = mInputReader.getRawData()
        
        # Training data initialization
        self.trainingData = np.array( np.empty_like( self.rawData ) )
                

    def setOutCSVPath( self , fPath ):
        '''@param fPath: relative location and name of feature dump CSV'''
        self.outCSVPath = fPath


    def getFeatures( self ):
        return self.features


    def getTrainingData( self ):
        return self.trainingData

    
    @abstractmethod
    def extractFeatures( self ):
        ''' This method is to be implemented by subclasses'''
        pass


    def writeFeaturesToCSV( self ):
        ''' 
        Dump the transformed data out to CSV for external eval
        Note: This shouldn't be called w/o extracting features from a 
        derived class first.
        '''
        mDumpFile = open( self.outCSVPath, 'w', newline='' )
        mCSVWriter = csv.writer( mDumpFile, delimiter=',' )
        
        # First write the features to the first row of the dump file
        mCSVWriter.writerow( self.features )

        # Then, dump all training data writing by row/record
        mCSVWriter.writerows( self.trainingData )

        # Release file i/o
        mDumpFile.close()


    def __del__( self ):
        ''' Destructor - No cleanup needed yet..'''
        pass
