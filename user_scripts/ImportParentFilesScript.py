# Imports required modules for adding parents
import os.path as path
import sys

# Adds parent directory
p = path.abspath('.')
sys.path.insert(2, p)
# Imports parent modules

def dummyFunc():
    pass