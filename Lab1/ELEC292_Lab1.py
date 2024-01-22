# Question 1
list_1 = [-1, 2, 3, 9, 0]
list_2 = [1, 2, 7, 10, 14]
print(list_1 + list_2)

# Question 2
list_1 = [-1, 2, 3, 9, 0]
list_2 = [1, 2, 7, 10, 14]
list_3 = []
for i, j in zip(list_1, list_2):
    list_3.append(i + j)
print(list_3)

# Question 3
import numpy as np
my_4d_array = np.array([[[[1], [2]], [[3], [4]], [[5], [6]]], [[[7], [8]], [[9], [10]], [[11], [12]]]])
print(my_4d_array.shape)

# Question 4
import numpy as np
my_3d_array = np.arange(27).reshape((3, 3, 3))
print(my_3d_array[:, :, 0])
print(my_3d_array[1, 1, :])
print(my_3d_array[:, 0:3:2, 0:3:2])

# Question 5
import numpy as np
my_3d_array = np.arange(27).reshape((3, 3, 3))
print(my_3d_array[[0, 1, 2], [1, 2, 0], [1, 2, 0]])
print(my_3d_array[1, [0, 2], [0, 2]])

# Question 6
import numpy as np
my_2d_array = np.arange(-10, 20).reshape((5, 6))
sum_cols = my_2d_array.sum(0)
indexing_array = sum_cols % 10 == 0
print(my_2d_array[:, indexing_array])
