# 16. Write a Python program to print the index of the character in a string.
def findIndex(str,ch):
    pos=-1
    try:
        pos=str.index(ch)
    except ValueError as v:
        pos = -1
    return pos

inputStr = input("Enter the string:\n")
ch = input("Enter a char:\n")
pos =findIndex(inputStr,ch)
if(pos!=-1):
    print(pos)
else:
    print(ch, "not found")