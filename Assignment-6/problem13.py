# 13. Take 10 integers from keyboard using loop and print their average value on the screen.

def calcAvg(nums):
    sum = 0
    for  x in nums:
        sum+=x
    avg = sum/len(nums)
    return avg

n = int(input("Enter the val of n:"))
nums=[]
for i in range(n):
    x = int(input())
    nums.append(x)
print("Average : ",round(calcAvg(nums),3))
