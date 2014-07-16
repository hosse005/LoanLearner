#!/usr/bin/python3
import csv

class InputReader:
    '''
    This class is responsible for reading the input resource
    into the program.  Class holds raw data read from the resource, and
    defers any further processing of data to futher classes.
    '''

    def __init__( self, fPath ):
        ''' 
        Constructor - arguments passed from main
        @param fPath: relative location and name of input resource
        '''
        self.__inputFilePath = fPath
        self.__rawData = list()
        
        # Attempt to access requested file
        try:
            self.__inputFile = open( self.__inputFilePath, 'r' )
            self.__reader = csv.reader( self.__inputFile, delimiter=',' )   
        except FileNotFoundError:
            print( "Couldn't open input file %s" % self.__inputFilePath )
            return
                                                        
    def setFilePath( self, fPath ):
        '''@param fPath: relative location and name of input resource'''
        self.__inputFilePath = fPath

    def readFile( self ):

        # Log status - TODO: move this to a logging class
        print( 'Reading input file..' )

        for row in self.__reader:
            self.__rawData.append( row )

    def getRawData( self ):
        return self.__rawData

    def __del__( self ):
        ''' Destructor - Close file connection '''
        try:
            self.__inputFile.close()
        except:
            pass
