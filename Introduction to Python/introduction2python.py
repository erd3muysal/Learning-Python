# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:57:45 2018

@author: R. Erdem Uysal
"""
# Pattern Recognition and Machine Learning

#%% Numpy module

# We like to call numpy briefly "np"
import numpy as np              

# Python list accepts any data types
v = [1, 2, 3, "hello", None]    

print(v[-1])

# Define a numpy array (vector)
v = np.array([1, 2, 3, 4])      

# Note: the above actually casts a
# Python list in to a numpy array.

# Resize into 2x2 matrix
V = np.resize(v, (2, 2))         

# Reshape doing same job with the resize method
V2 = v.reshape(2, 2)

# Invert:
np.linalg.inv(V)

#%% More on Vectors

# Create a vector from 1 to 10 with 0.5 step
np.arange(1, 10, 0.5) # Arguments: (start, end, step)

# Note that the endpoint is not included (unlike Matlab).

# Create a vector from 1 to 10 with equal difference
np.linspace(1, 10, 5) # Arguments: (start, end, num_items)

# Create a identity matrix
np.eye(3)

# Create random  vector
np.random.random(3)

# Create random matrix
np.random.randn(2, 3)

#%% Matrices

# A matrix is defined similarly
# either by specifying the values manually,
# or using special functions.

# A matrix is simply an array of arrays
# May seem complicated at irst, but is in fact
# nice for N-D arrays.

np.array([[1, 2], [3, 4]])

from scipy.linalg import toeplitz, hilbert # You could also "...import *"

toeplitz([3, 1, -1])

hilbert(3)

#%% Matrix Product

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A*B # Elementwise product (Matlab: A.*B)

np.dot(A, B) # Matrix product; alternatively: np.matmul

# With Numpy version 1.10+ and
# Python 3.5+, matrix multiplication can
# be done with the @ operator:
A@B 

#%% Indexing
x = np.arange(1, 11)
x[0:8:2] # Unlike Matlab indexing starts from 0

# Note: use square brackets for indexing
# Note2: colon opertator has the order start:end:step;
# not start:Step:end as in Matlab

x[5:]          # All items from the 5'th
x[:5]          # All items until the 5'th
x[::3]         # All items with step 3

x[-3:]             # Three last items
x[::-1]            # Items in inverse order

# Also matrices can be indexed similarly.
# This operation is called slicing, and
# the result is a slice of the matrix.

M = np.reshape(np.arange(0, 36), (6, 6))

# Here we request for items on the rows 2:4 = [2, 3]
# and columns 1, 2, 4
# Note, that with matrices, the first index is the row;
# not "x-coordinate". This order is called " Fortran style"
# or "column major" while the alternative is "C style" or "row major".
M[2:4, [1, 2, 4]]


# To specify only column or row induces, use ":" alone.

# Now we wish to extract two bottom rows.
# M[4:, : ] reads "give me all rows after the 4th and all columns".
# Alternative forms would be, e.g., M[-2:, :] and M[[4, 5], :]
M[4:, :]  

#%% N-Dimensional arrays

# Generate a random "image" array:
A = np.random.rand(1000, 3, 96, 128)

# What size is it?
A.shape

# Here, dimensions are: image index, color channel, y-coordinate, x-coordinate.
# Sometimes, a shorter name is used: "(b, c, 0, 1) order".

# Access the pixel (4, 3) of 2nd color channel
# of the 2nd image
A[1, 2, 3, 4]

# Request all color channels:
A[1, :, 3, 4]

# Request a comple 96x128 color channel:
A[1, 2, :, :]

# Equivalent shorter notation:
A[1, 2, ...]

#%% Functions

# Define our first function
def hello(target):
    print("Hello " + target + "!")
    
hello("world")
hello("Finland")

# We can also define the default argument:
def hello(target = "world"):
    print("Hello " + target + "!")
    
hello()
hello("Finland")

# One can also assign using the name
hello(target = "Finland")

# Functions can be imported to orher files using import.
# Functions arguments can be positional or named
# Named arguments improve readability and are handy
# for setting the last argument in a long
#%% Loops and Stuff

for lang in ['Assembler', 'Python', "Matlab", 'C++']:
    if lang in ['Assembler', "C++"]:
        print("I am ok with %s" % (lang))
    else:
        print("I love %s" % (lang))
        
# Read all lines of a file until the end
        
fp = open("myfile.txt", "r")
lines = []

while True:
    
    try:
        line = fp.readline()
        line.append(line)
    except:
        # File ended
        break
    
fp.close()

# For can loop over anything iterable, such as a list or a file.
# In Matlab, appending values to a vector in a loop,
# is not recommended. Python lists are actual lists,
# so appending is fine

#%% Example: Reading in a Data File

import nump as np

if __name__ == "__main__":
    
    X = []   # Rows of the file go here
    
    # We use Python's with statement
    # Then we do not have to worry
    # about closing it.
    
    with open("ovarian.csv", "r") as fp:
        
        # File is iterable, so we can
        # read it directly (instead of
        # using readline).
        
        for line in fp:
            
            # Skip the first line:
            if "Sample_ID" in line:
                continue
            
            # Otherwise, split the line
            # to numbers:
            values = line.split(";")
            
            # Omit the first item
            # ("S1" or similar):
            values = values[1:]
            
            # Cast each item from
            # string to float:
            values = [float(v) for v in values]
            
            # Appent to X
            X.appent(values)
        # Now, X is a list of lists. Cast to
        # Numpy array:
        X = np.array(X)
        
        print("All data read.")
        print("Result size is %s" %(str(X.shape)))

#%% Visualization

import matplotlib.pyplot as plt
import numpy as np

N = 100
n = np.arange(N)   # Vector [0, 1, 2, ..., N-1]
x = np.cos(2 * np.pi * n * 0.03)
x_noisy = x + 0.2 * np.random.randn(N)
    
fig = plt.figure(figsize = [10, 5])

plt.plot(n, x, 'r-',
         linewidth = 2,
         label = "Clean Sinusoid")

plt.plot(n, x_noisy, 'bo-',
         markerfacecolor = "green",
         label = "Noisy Sinusoid")

plt.grid("on")
plt.xlabel("Time in $\mu$s")
plt.ylabel("Amplitude")
plt.title("An example Plot")
plt.legend(loc = "upper left")

plt.show()
plt.savefig("C:\Users\ASUS\Downloads\Courses\Intro to Python for Data Science/intermediate_python_ch1_slides.pdf",
            bbox_inches = "tight")