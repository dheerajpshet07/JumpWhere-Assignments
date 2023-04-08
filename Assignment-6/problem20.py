# 20.You are given with a list of integer elements. Make a new list which will store square of elements of previous list.
def computeSquare(listOfNum,sqrOfNum):
    for x in listOfNum:
        sqrOfNum.append(x*x)

n = int(input("Enter the val of n:"))
listOfNum = []
sqrOfNum = []
for i in range(n):
    x = int(input())
    listOfNum.append(x)
computeSquare(listOfNum,sqrOfNum)

print("Square of numbers : ",sqrOfNum)

