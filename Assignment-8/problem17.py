# 7) Write a Python program to sort a given matrix in ascending order according to the sum of its rows using lambda. 
# Original Matrix: [[1, 2, 3], [2, 4, 5], [1, 1, 1]] 
# Sort the said matrix in ascending order according to the sum of its rows [[1, 1, 1], [1, 2, 3], [2, 4, 5]]
#  Original Matrix: [[1, 2, 3], [-2, 4, -5], [1, -1, 1]] 
# Sort the said matrix in ascending order according to the sum of its rows [[-2, 4, -5], [1, -1, 1], [1, 2, 3]] 

matrix_1 = [[1, 2, 3], [2, 4, 5], [1, 1, 1]] 
matrix_2 = [[1, 2, 3], [-2, 4, -5], [1, -1, 1]]
sort_matrix = lambda matrix:sorted(matrix, key = lambda x:sum(x))
print("Matrix1  -- Sorted 2d matrix according to their sum : \n",sort_matrix(matrix_1) )
print("\nMatrix2  -- Sorted 2d matrix according to their sum : \n",sort_matrix(matrix_2) )