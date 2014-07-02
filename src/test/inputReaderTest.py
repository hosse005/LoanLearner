#!/usr/bin/python3

import sys
sys.path.append( '..' )
from inputReader import InputReader
import unittest

# Test resource must be relative to class under test
testFile = '../../res/InputReaderTest.csv'

class InputReaderTest(unittest.TestCase):

    def setUp( self ):
        ''' Construct our InputReader object, pass it the test csv file '''
        self.mInputReader = InputReader( testFile )

    def test_read( self ):
        ''' Test the csv read functionality against known values '''
        self.mInputReader.readFile()
        mRawData = self.mInputReader.getRawData()
        self.assertEqual( mRawData, [['InputReader', 'Test', 'CSV'],
                                    ['1', '44', '-4.3'],
                                     ['234', '-45', '0.45']] )

    def test_pathSet( self ):
        ''' Test set file path and FileNotFoundError exception'''
        # Set a bogus file name
        self.mInputReader.setFilePath( '../../res/NonExist.csv' )
        self.assertRaises( FileNotFoundError, self.mInputReader.readFile() )

if __name__ == '__main__':
    unittest.main()
