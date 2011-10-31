'''
Created on Oct 31, 2011

@author: joel
'''
import pypwdg.parallel.decorate as ppd

@ppd.parallel(lambda n: lambda somelist: ppd.partitionlist(n,somelist))
def myexpensivefn(somelist):
    return sum([x**2 for x in somelist])

import pypwdg.parallel.main

if __name__ == "__main__":
    print myexpensivefn(range(100))