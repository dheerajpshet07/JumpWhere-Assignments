# 7. Write a Python program that counts the number of elements within a list that are greater than 30.

def countGreaterThan30(numbers):
    cnt=0
    for x in numbers:
        if x > 30:
            cnt+=1
    return cnt

n = int(input("Enter the val of n:"))
nums = []
print("Enter ",n, " values:")
for i in range(n):
    x = int(input())
    nums.append(x)
print("Number of element greater than 30 is : ",countGreaterThan30(nums))