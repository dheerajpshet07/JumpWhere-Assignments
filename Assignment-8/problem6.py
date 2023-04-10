# 6) Write a Python class to implement pow(x, n) 

class Solution:
    def pow(self,x,n):
        res = 1.0
        nn = n
        if n<0:
            nn = -1.0*n

        while(nn>0):
            if(nn%2 == 1):
                res = res*x
                nn = nn-1
            else:
                x = x*x
                nn = nn/2.0
        
        if(n < 0):
            res = float(1.0)/res
        return res

obj = Solution()
x = float(input("Enter the value of x\n"))
n = int(input("Enter the value of n\n"))
print("Pow(",x,",",n,")    : ",obj.pow(x,n))


