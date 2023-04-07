# 7. Write a Python program to find the first appearance of the substring 'not' and 'poor' from a given string, if 'not' follows the 'poor', replace the whole 'not'...'poor' substring with 'good'. Return the resulting string.
# Sample String : 'The lyrics is not that poor!'
# 'The lyrics is poor!'
# Expected Result : 'The lyrics is good!'
# 'The lyrics is poor!'


def stringOperation(inputStr):
    notIndex = inputStr.find('not')
    poorIndex = inputStr.find('poor')
    outputStr=''
    if(notIndex == -1 or poorIndex == -1):
        outputStr = inputStr
    else:
        outputStr = inputStr[:notIndex]+"good"+inputStr[poorIndex+4:]
    return outputStr


inputStr = input("Enter the string:\n")
print(stringOperation(inputStr))

