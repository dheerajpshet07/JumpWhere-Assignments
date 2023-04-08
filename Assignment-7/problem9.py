# 9. Write a Python program to multiply all the items in a dictionary.


def calcProd(dict):
    listOfNums = []
    for val in dict:
        listOfNums.append(dict[val])

    res = 1
    for x in dict.values():
        res *= x
    return res

dict = {'a': 100, 'b': 200, 'c': 300}
print("Sum :", calcProd(dict))