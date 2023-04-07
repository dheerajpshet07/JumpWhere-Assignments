# 12. Write a Python function to convert a given string to all uppercase if it contains at least 2 uppercase characters in the first 4 characters.

def stringOperation(str):
    upCnt=0
    for i in range(4):
       if(str[i].isupper()):
           upCnt+=1
    if(upCnt>=2):
        return str.upper()
    else:
        return str


str = input("Enter a string:")
print(stringOperation(str))
