'''
Created on Dec 20, 2010

@author: joel
'''
import unittest
import pypwdg
import sys

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(".")
    unittest.TextTestRunner(verbosity=2).run(suite)