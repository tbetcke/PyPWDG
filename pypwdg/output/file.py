import string        
import pypwdg.output.solution as pos

class ErrorFileOutput(object):
    ''' A simple utility class that outputs a set of errors as a python-readable(ish) array'''
    
    def __init__(self, name, ns, g, bounds, npoints):
        self.ftxt = open(name + ".txt", 'a')
        self.ftxt.write(name.translate(None, string.punctuation + string.whitespace)+' = (')
        self.ftxt.write(str(ns)+", [")
        self.bounds = bounds
        self.npoints = npoints
        self.g = g
        self.docomma = False
    
    def process(self, n, solution):
        err = pos.comparetrue(self.bounds, self.npoints, self.g, solution)
        print n, err
        if self.docomma: self.ftxt.write(', ')
        self.docomma = True
        self.ftxt.write(str(err))
        
    def __del__(self):
        self.ftxt.write('])\n')
