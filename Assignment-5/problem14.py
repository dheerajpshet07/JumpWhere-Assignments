# Write a Python program to print the following floating numbers upto 2 decimal places.
# 3.1415926
testCase = int(input("Enter how many numbers you want to check "))
while(testCase):
    num = float(input("enter a floating point number:  "))
    print(round(num,2))
    testCase-=1

    