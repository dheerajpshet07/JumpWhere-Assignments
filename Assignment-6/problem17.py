# 17. Using range(1,101), make three list, 
# one containing all even numbers
# one containing all odd numbers 
# One containing only prime numbers..

import math
prime = [True for i in range(102)]
prime[0] = False
prime[1] = False

for i in range(2,int(math.sqrt(102))):
    if prime[i] == True:
        for j in range(i*i,102,i):
            prime[j] = False



def isPrime(num):
    return prime[num]

primeNum = []
evenNum = []
oddNum = []

for p in range(1,102):
    if(isPrime(p)):
        primeNum.append(p)
    if(p%2 == 0):
        evenNum.append(p)
    else:
        oddNum.append(p)

print("Even Numbers :\n",evenNum)
print("Odd Numbers :\n",oddNum)
print("Prime Numbers :\n",primeNum)



    
