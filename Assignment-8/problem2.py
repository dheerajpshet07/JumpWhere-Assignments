# 2) Write a Python class to find validity of a string of parentheses, '(', ')', '{', '}', '[' and ']. These brackets must be close in the correct order, for example "()" and "()[]{}" are valid but "[)", "({[)]" and "{{{" are invalid.

class Solution:
    def checkBracketValidity(self, str):
        stack = []
        for x in str:
            if x in ("(","{","["):
                stack.append(x)
            elif len(stack)>0:
                if x == ")":
                    if(stack[-1] == "("):
                        stack.pop()
                elif x == "}":
                    if(stack[-1] == "{"):
                        stack.pop()
                elif x == "]":
                    if(stack[-1] == "["):
                        stack.pop()
            else:
                return "Invalid"
        
        if(len(stack)):
            return "Invalid"
        else:
            return "valid"

obj = Solution()
str = input("Enter the string to check validity:  ")
print(obj.checkBracketValidity(str))

        
        
