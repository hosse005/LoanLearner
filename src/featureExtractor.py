#!/usr/bin/python3

from abc import ABCMeta
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
        self.__rawData = list()
        self.__features = list()
        self.__trainingData = np.array()

        # Feature dump path
        self.__outCSVPath = '../../res/featureDump.csv'
        
        # Get raw data from the passed InputReader
        self.__rawData = mInputReader.getRawData()
        

    def setOutCSVPath( self , fPath ):
        '''@param fPath: relative location and name of feature dump CSV'''
        self.__outCSVPath = fPath


    @abstractmethod
    def getFeatures( self ):
        ''' Derived classes should return extracted feature list'''
        pass


    @abstractmethod
    def getTrainingData( self ):
        ''' Derived classes should return extracted training data'''
        pass


    def writeFeaturesToCSV( self ):
        ''' 
        Dump the transformed data out to CSV for external eval
        Note: This shouldn't be called w/o extracting features from a 
        derived class first.
        '''
        mDumpFile = open( self.__outCSVPath, 'w', newline='' )
        mCSVWriter = csv.writer( mDumpFile, delimiter=',' )
        
        # First write the features to the first row of the dump file
        mCSVWriter.writeline( self.__features )

        # Then, dump all training data writing by row/record
        mCSVWriter.writerows( self.__trainingData )

        # Release file i/o
        mDumpFile.close()


    def __del__( self ):
        ''' Destructor - No cleanup needed yet..'''
        pass
