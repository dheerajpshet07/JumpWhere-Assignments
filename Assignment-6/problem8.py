# 8. Take values of length and breadth of a rectangle from user and check if it is square or not.

def isItsquare(length,breadth):
    if(length == breadth):
        return True
    else:
        return False

length = int(input("Enter the val of length:"))
breadth = int(input("Enter the val of breadth:"))
print(isItsquare(length,breadth))