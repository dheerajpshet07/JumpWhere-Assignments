# 7) Write a Python class to reverse a string word by word. 
# Input string : 'hello .py' Expected Output : '.py hello' 
class Solution:
    def reverseSentence(self,str):
        words = str.split(" ")
        words = words[::-1]
        revSentence = " ".join(words)
        return revSentence

obj = Solution()
str = input("Enter the string :\n")
print("Sentence in reverse order : ", obj.reverseSentence(str))