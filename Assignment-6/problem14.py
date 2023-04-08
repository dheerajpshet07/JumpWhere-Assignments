# 14. Print multiplication table of 24, 50 and 29 using loop.

def printMultiTable(num):
    for i in range(1,11):
        print(num," * ",i," = ",num*i)

n = int(input("Enter the number for which you want multi table: "))
printMultiTable(n)
