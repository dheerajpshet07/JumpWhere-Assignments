# 10) Write a Python program to sort a given mixed list of integers and strings using lambda. Numbers must be sorted before strings. 
# Original list: [19, 'red', 12, 'green', 'blue', 10, 'white', 'green', 1] 
# Sort the said mixed list of integers and strings: [1, 10, 12, 19, 'blue', 'green', 'green', 'red', 'white']

my_list = [19, 'red', 12, 'green', 'blue', 10, 'white', 'green', 1]

result = sorted(my_list, key = lambda x:(isinstance(x,str),x))

print("Original list:", my_list)
print("Sort the said mixed list of integers and strings:", result)
