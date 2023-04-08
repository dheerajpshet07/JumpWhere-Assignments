# 11. Write a Python program to sort a dictionary by key.

my_dict = { 'num6': 6, 'num3': 3, 'num2': 2, 'num4': 4, 'num1': 1, 'num5': 5}
print("Before sorting  ", my_dict)
listOfKeys = list(my_dict.keys())
listOfKeys.sort()
sorted_dict = {i:my_dict[i] for i in listOfKeys}
print("After sorting  ", sorted_dict)