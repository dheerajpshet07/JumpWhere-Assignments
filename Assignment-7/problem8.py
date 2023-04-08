# 8.  Write a Python program to sum all the items in a dictionary.

def calcSum(dict):
    listOfNums = []
    for val in dict:
        listOfNums.append(dict[val])

    res = sum(listOfNums)
    return res

dict = {'a': 100, 'b': 200, 'c': 300}
print("Sum :", calcSum(dict))