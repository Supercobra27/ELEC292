# Author(s): Ryan Silverberg
# Purpose: LAB1 of data science

import pandas as ps, numpy as np

# Question 1
list_1 = [-1, 2, 3, 9, 0]
list_2 = [1, 2, 7, 10, 14]
print((list_1 + list_2))

my_list = [1, 2, 3, 4, 5]
print('\n')
for x in my_list:
    print(x)

print('\n')
#Question 2
    
for i in range(0, len(list_1)):
    list_1[i] = list_1[i]+list_2[i]
    print(list_1[i])

print('\n')
print(list_1)

print('\n')

# Question 3

# Given Arrays
array1D = np.array([1, 2, 3, 4, 5])
print(array1D.shape)
print('\n')
array2D = np.array([[1, 2, 3], [4, 5, 6]])
print(array2D.shape)
print('\n')
array3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(array3D.shape)
print('\n')

# 4D Array
array4D = np.array([[[[1],[7]],[[4],[7]],[[1],[2]]],[[[1],[7]],[[4],[7]],[[1],[2]]]])
print(array4D.shape)
print('\n')

# Question 4 [Searching Arrays]
 # first is choosing which matrix, then columns, then rows
arrangedarray = np.arange(27).reshape(3,3,3)
print(arrangedarray[:,:,0]) # print first row of all arrays
print('\n')
print(arrangedarray[1,1,:]) # print middle array middle row
print('\n')
print(arrangedarray[:,0::2,0::2]) # print all corners

# Question 5 [Selecting Cells in Arrays]
selectarray = np.arange(27).reshape(3,3,3)
print(selectarray[0:1:2,1:2:0,1:2:0])
print('\n')
print(selectarray[[1,1],[0,2],[0,2]])
print('\n')

# Question 6
indexingarray = np.arange(-10,20).reshape(5,6)
print(indexingarray.size) # Also a thing
print('\n')

# axis 0 is columns
boolarray = indexingarray.sum(axis=0) % 10 == 0 # condition against each column item

print(indexingarray[:, boolarray]) # will use the column if the boolarray has true

# Summary
# make arrays using np.arange(start, end).reshape(dimensions)
# use boolean conditions to select specific rows based on conditions and then print those based on that
# use array.shape, array.size, array[:,...] to print where : will print all cols in that dimension
# to search for specific cells, i.e. 2D array array[[0],[0]] will get the item at (0,0)