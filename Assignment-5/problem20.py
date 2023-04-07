# 20. Write a Python program to remove all consecutive duplicates of a given string.


def removeConsecutiveDuplicates(inputStr):
    opStr = inputStr[0]
    for i in range(1,len(inputStr)):
        if inputStr[i-1] != inputStr[i]:
            opStr += inputStr[i]
    
    return opStr
        
inputStr = input("Enter the string:\n")
print("Original string:" + inputStr)
print("After removing consecutive duplicates: ",removeConsecutiveDuplicates(inputStr))
