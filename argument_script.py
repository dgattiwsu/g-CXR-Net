#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:33:31 2021

@author: dgatti
"""

#!/usr/bin/python

import sys, getopt, argparse

# In[2]:
# def main(argv):
#    inputfile = ''
#    outputfile = ''
#    try:
#       opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
#    except getopt.GetoptError:
#       print('test.py -i <inputfile> -o <outputfile>')
#       sys.exit(2)
#    for opt, arg in opts:
#       if opt == '-h':
#          print('test.py -i <inputfile> -o <outputfile>')
#          sys.exit()
#       elif opt in ("-i", "--ifile"):
#          inputfile = arg
#       elif opt in ("-o", "--ofile"):
#          outputfile = arg
#    print('Input file is "', inputfile)
#    print('Output file is "', outputfile)

# if __name__ == "__main__":
#    main(sys.argv[1:])

# In[3]:
# Initialize parser
parser = argparse.ArgumentParser()
parser.parse_args()
