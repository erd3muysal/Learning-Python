# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 16:30:57 2018

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    
    # Question 1: load data using numpy
    
    location_data = np.loadtxt("locationData.csv")
    print(np.shape(location_data))
    
    # Question 2: visualize using plt functions
    
    plt.plot(location_data[:, 0], location_data[:, 1], 'b-')
    plt.shoy()
    
    ax = plt.subplot(1, 1, 1, projection = "3d")
    plot3D = ax.plot(location_data[:, 0],
                     location_data[:, 1],
                     location_data[:, 2])
    
    plt.show()
    
    # Question 3: Normalize data to zero mean and unit variance
    
    def normalize_data(data):
        
        return (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)
    
    X_norm = normalize_data(location_data)
    
    print(np.mean(X_norm, axis = 0))   # Should be [0, 0, 0]
    print(np.std(X_norm, axis = 0))    # Should be [1, 1, 1]