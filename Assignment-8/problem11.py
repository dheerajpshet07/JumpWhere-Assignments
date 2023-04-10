# 1) Write a Python program to create a lambda function that adds 15 to a given number passed in as an argument, also create a lambda function that multiplies argument x with argument y and print the result.
#  Sample Output: 25 48 


addFifteen = lambda x:x+15
multiXY = lambda x,y : x*y

x = int(input("Enter the value of x:\n"))
y = int(input("Enter the value of y:\n"))
print("Add 15 : ", addFifteen(x))
print("Multiply x,y : ", multiXY(x,y))

