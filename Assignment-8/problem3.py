# 3) Write a Python class to get all possible unique subsets from a set of distinct integers Input : [4, 5, 6] Output : [[], [6], [5], [5, 6], [4], [4, 6], [4, 5], [4, 5, 6]] 
import copy
class Solution:
    def generatePowerSet(self, arr):
        n = len(arr)
        pwrSet = []
        for i in range(pow(2,n)):
            st =[]
            for j in range(n):
                if(i & (1<<j)):
                    st.append(arr[j])
            pwrSet.append(st)
            st = []
        return pwrSet

    def generateSubset(self,arr,n,i,pwrSet):
        temp =[]
        for i in range(i,n):
            temp.append(arr[i])
            temp1 = copy.deepcopy(temp)
            pwrSet.append(temp1)

        


    def recGeneratePowerSet(self, arr):
        n = len(arr)
        pwrSet = []
        pwrSet.append([])
        for i in range(0,n):
            self.generateSubset(arr,n,i,pwrSet)
        return pwrSet




obj = Solution()
n = int(input("Enter the number of values: \n"))
arr = []
for i in range(n):
    x = int(input())
    arr.append(x)
print(obj.recGeneratePowerSet(arr))