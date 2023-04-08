# 18. Write a Python program to check if all dictionaries in a list are empty or not.
# Sample list : [{},{},{}]
# Return value : True
# Sample list : [{1,2},{},{}]
# Return value : False


def isDictEmpty(my_list):
    res = True
    for x in my_list:
        if bool(x):
            res = False
            break
    if res == True:
        return "All dict in list are empty"
    else:
        return "All dict in list are not empty"

l1 = [{},{},{}]
l2 = [{1,2},{},{}]

print(isDictEmpty(l1))
print(isDictEmpty(l2))