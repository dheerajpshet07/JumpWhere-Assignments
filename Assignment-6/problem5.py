# 5. Write a Python program to calculate the sum and average of n integer numbers (input from the user). Input 0 to finish

def findSumAndAvg(finish):
    sum = int((finish*(finish+1))/2)
    avg = sum/finish
    return sum,avg


n = int(input("Enter the val of n:"))
sum,avg = findSumAndAvg(n)
print("Sum : ",sum,"\nAvg : ",avg)