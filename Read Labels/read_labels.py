# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 15:41:31 2018

@author: ASUS
"""

import numpy as np

if __name__ == "__main__":         
# Where it checks that if i am actually executing this
# code or i am importing some function from this file.
# This is not necessery but i like it. I thing it is also makes things more regulable
# Now I can see from here. OK this is where the main file starts.
    
    f = open("y_train.csv", "r")   # It is quite similar what we have in C++. "r" represents read.
    
    labels = []
    
    for line in f:
        
        if "id" in line:     # First we check, if I am on the first line because it is not hold any information.
            continue         # I do not read this line, otherwise read.
        
        idx, label = line.split(",")   # Splits with respect to ","
        labels.append(label.strip())   # Strip removes spaces between strings.
        
    print(labels)
    