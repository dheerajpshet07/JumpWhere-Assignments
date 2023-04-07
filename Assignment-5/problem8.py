# 8. Write a Python function that takes a list of words and returns the length of the longest one. 
def findLongestWordLength(inputStr):
    list = inputStr.split(" ")
    longestLen = -1
    for word in list:
        if(len(word) > longestLen):
            longestLen = len(word)
    return longestLen
   


inputStr = input("Enter the string:\n")
print(findLongestWordLength(inputStr))