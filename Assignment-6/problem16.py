# 16. Take inputs from user to make a list. Again take one input from user and search it in the list and delete that element, if found. Iterate over list using for loop.

n = int(input("Enter the val of n:"))
nums=[]
for i in range(n):
    x = int(input())
    nums.append(x)

searchElem = int(input("Enter the element to search and delete:"))
try:
    while True:
        nums.remove(searchElem)
except ValueError:
    pass

print("After deleting  :  ",nums)