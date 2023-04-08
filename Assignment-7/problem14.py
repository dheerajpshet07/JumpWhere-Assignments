# 14. Write a Python program to check a dictionary is empty or not. 

def isItEmpty(x):
    if bool(x):
        return "dictionary is not empty"
    else:
        return "dictionary is empty"


my_dict = {'a': 100, 'b': 200, 'c': 300}
temp = {}

print(isItEmpty(my_dict))
print(isItEmpty(temp))

