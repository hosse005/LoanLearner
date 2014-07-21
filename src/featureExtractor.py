#!/usr/bin/python3

from abc import ABCMeta, abstractmethod
from inputReader import InputReader
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
        # Feature dump and filter path
        self.outCSVPath = '../../tmp/featureDump.csv'
        self.filterCSVPath = '../../res/FeatureFilter.csv'

        # Get raw data from the passed InputReader
        mInputReader.readFile()
        self.rawData = mInputReader.getRawData()

        # Initialize feature set and training data from raw data
        self.features = self.rawData[0]
        self.trainingData = np.array( self.rawData[1:] )
        
        # Construct the InputReader used for feature filtering
        self.filterReader = InputReader( self.filterCSVPath )

        # Initialize number of samples removed
        self.nRmvSamples = 0


    def setOutCSVPath( self , fPath ):
        '''@param fPath: relative location and name of feature dump CSV'''
        self.outCSVPath = fPath


    def setFilterCSVPath( self , fPath ):
        '''@param fPath: relative location and name of feature filter CSV'''
        self.filterCSVPath = fPath


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
        

    def applyFeatureFilter( self ):
        ''' 
        Reads the filter resource file and accordingly removes the feature
        from each sample.
        '''
        # Read out the resource content
        filterReader.readFile()

        # Stash the results to a local list
        mFilterList = filterReader.getRawData()
        
        # Use our list index method to find appropriate column in feature 
        # list to remove
        for feature in mFilterList:
            try:
                idx = listIdx( feature )
                del self.features[idx]
                self.trainingData = np.delete( self.trainingData, idx, 1 )
            except ValueError:
                print( 'Unable to remove feature %s!' % feature )
    

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


    @abstractmethod
    def extractFeatures( self ):
        ''' This method is to be implemented by subclasses'''
        pass


    def __del__( self ):
        '''No Destructor implementation'''
        pass
