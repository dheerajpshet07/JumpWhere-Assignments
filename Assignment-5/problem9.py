# 9. Write a Python program to remove the nth index character from a nonempty string.
def removeNthChar(inputStr,pos):
        outputStr = inputStr[:pos]+inputStr[pos+1:]
        return outputStr


inputStr = input("Enter the string:\t")
pos = int(input("Enter the pos:\t"))
print(removeNthChar(inputStr,pos))

