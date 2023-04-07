# 4. Write a Python program to get a string from a given string where all occurrences of its first char have been changed to '$', except the first char itself.
# Sample String : 'restart'
# Expected Result : 'resta$t'
str = input("Enter the string")
isR = False
finalString = ''
for s in str:
    if s == 'r':
        if isR == True:
            finalString+='$'
        else:
            isR = True
            finalString+=s
    
    else:
        finalString+=s

print(finalString)


