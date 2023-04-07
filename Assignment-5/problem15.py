# 15. Write a Python program to count repeated characters in a string.
# Sample string: thequickbrownfoxjumpsoverthelazydog
# Expected output :
# o 4
# e 3
# u 2
# h 2
# r 2
# t 2


def countChar(inputStr):
    dict = {}
    for ch in inputStr:
        if ch in dict:
            dict[ch]+=1
        else:
            dict[ch]=1
    
    
    return dict

    
   


inputStr = input("Enter the string:\n")
print(countChar(inputStr))