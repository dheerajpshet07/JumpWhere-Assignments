# 8. Write a Python program to swap comma and dot in a string.
# Sample string: "32.054,23"
# Expected Output: "32,054.23"
def swapChars(str):
    dotPos = str.index('.')
    commaPos = str.index(',')
    opString=""
    if(dotPos<commaPos):
        opString = str[:dotPos]+","+str[dotPos+1:commaPos]+"."+str[commaPos+1:]
    else:
        opString = str[:commaPos]+"."+str[commaPos+1:dotPos]+","+str[dotPos+1:]

    return opString

inputStr = input("Enter the string:\n")
print(swapChars(inputStr))
