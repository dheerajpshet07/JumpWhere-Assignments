# 5) Write a Python class to find the three elements that sum to zero from a set of n real numbers. Input array : [-25, -10, -7, -3, 2, 4, 8, 10] Output : [[-10, 2, 8], [-7, -3, 10]] 

class Solution:
    def threeSum(self, numList):
        unique3 = []
        numList.sort()
        for i in range(len(numList)-2):
            if (i == 0 or (i>0 and numList[i]!=numList[i-1])):
                low = i+1
                high = len(numList)-1
                while(low < high):
                    sum = numList[low] + numList[high] + numList[i]
                    if sum == 0:
                        sm=[]
                        sm.append(numList[i])
                        sm.append(numList[low])
                        sm.append(numList[high])
                        unique3.append(sm)

                        while( low<high and numList[low] == numList[low+1]):
                            low += 1 

                        while( low<high and numList[high] == numList[high-1]):
                            high -= 1 
                        low+=1
                        high-=1
                    elif sum > 0:
                        high -= 1
                    else:
                        low += 1
        return unique3

obj = Solution()
n = int(input("Enter the num of inputs:\n"))
arr = []
for i in range(n):
    x = int(input())
    arr.append(x)

print("Test")
print("The unique subsets that sums upto 0 is : ", obj.threeSum(arr))