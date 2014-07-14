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
        # Feature dump path
        self.outCSVPath = '../../tmp/featureDump.csv'
        
        # Get raw data from the passed InputReader
        mInputReader.readFile()
        self.rawData = mInputReader.getRawData()

        # Initialize feature set and training data from raw data
        self.features = self.rawData[0]
        self.trainingData = np.array( self.rawData[1:] )
        
        # Initialize number of samples removed
        self.nRmvSamples = 0

    def setOutCSVPath( self , fPath ):
        '''@param fPath: relative location and name of feature dump CSV'''
        self.outCSVPath = fPath


    def getFeatures( self ):
        return self.features


    def getTrainingData( self ):
        return self.trainingData


    def setTrainingData( self, data ):
        assert( isinstance( data, np.ndarray ) )
        self.trainingData = data

    
    def getSampleCnt( self ):
        return len( self.trainingData )


    def getRmvSampleCnt( self ):
        return self.nRmvSamples


    def listIdx( self, feature ):
        '''
        Return the list index of a given feature
        @param feature: training feature
        @return index: index of passed feature
        '''
        return self.features.index( feature )

    
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
        '''No Destructor implementation'''
        pass
