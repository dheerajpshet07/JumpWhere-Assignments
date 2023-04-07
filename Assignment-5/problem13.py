# 13. Write a Python program to check whether a string starts with specified characters.

def stringOperation(str,ch):
    if str[0] == ch.upper() or str[0] == ch.lower():
        return True
    else:
        return False


str = input("Enter a string:")
ch = input("Enter the character:")
print(stringOperation(str,ch))
