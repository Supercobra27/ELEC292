# Question 6
import numpy as np
my_2d_array = np.arange(-10, 20).reshape((5, 6))
sum_cols = my_2d_array.sum(0)
indexing_array = sum_cols % 10 == 0
print(my_2d_array[:, indexing_array])
