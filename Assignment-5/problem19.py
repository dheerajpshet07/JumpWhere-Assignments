# 19. Write a Python program to find smallest and largest word in a given string.

def findLongestWord(inputStr):
    list = inputStr.split(" ")
    longestLen = -1
    longestWord=""
    for word in list:
        if(len(word) > longestLen):
            longestLen = len(word)
            longestWord = word
    return longestWord
   

def findShortestWord(inputStr):
    list = inputStr.split(" ")
    shortestLen = 1000
    shortestWord=""
    for word in list:
        if(len(word) < shortestLen):
            shortestLen = len(word)
            shortestWord = word
    return shortestWord


inputStr = input("Enter the string:\n")
print("Longest word ",findLongestWord(inputStr))
print("Shortest word ",findShortestWord(inputStr))
