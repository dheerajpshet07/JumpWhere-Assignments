# 10. Write a Python program that accepts a comma separated sequence of words as input and prints the unique words in sorted form (alphanumerically). 
# Sample Words : red, white, black, red, green, black
# Expected Result : black, green, red, white

def findUniqueueWords(inputStr):
    uniqWords= set()
    listW = inputStr.split(",")
    for word in listW:
        uniqWords.add(word)
    uniqWordsList = list(uniqWords)
    uniqWordsList.sort()
    listToStr = ','.join(map(str, uniqWordsList))
    return listToStr
    
   


inputStr = input("Enter the string:\n")
print(findUniqueueWords(inputStr))

