# 11. Write a Python function to reverses a string if it's length is a multiple of 4. def findUniqueueWords(inputStr):

def stringOperation(str):
    if(lenOfStr % 4 == 0):
        return str[::-1]
    else:
        return str


str = input("Enter a string:")
lenOfStr = len(str)
print(stringOperation(str))
