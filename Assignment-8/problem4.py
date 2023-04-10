# 4) Write a Python class to find a pair of elements (indices of the two numbers) from a given array whose sum equals a specific target number. Note: There will be one solution for each input and do not use the same element twice. Input: numbers= [90, 20,10,40,50,60,70], target=50 Output: 3, 4 


class Solution:
    def twoSum(self, arr, target):
        d ={}
        for x in arr:
            compliment = target - x
            if compliment in d:
                return [arr.index(compliment),arr.index(x)]
            d[x] = compliment

obj = Solution()
n = int(input("Enter the num of inputs:\n"))
arr = []
for i in range(n):
    x = int(input())
    arr.append(x)
target = int(input("Enter the target:\n"))
print(obj.twoSum(arr,target))




