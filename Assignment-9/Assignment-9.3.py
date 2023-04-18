# A balanced delimiter starts with an opening character ((, [, {), ends with a matching closing character (), ], } respectively), and has only other matching delimiters in between. A balanced delimiter may contain any number of balanced delimiters.

# Examples
# The following are examples of balanced delimiter strings:

# ()[]{}
# ([{}])
# ([]{})
# The following are examples of invalid strings:
# ([)]
# ([]
# [])
# ([})
# Input is provided as a single string. Your output should be True or False according to whether the string is balanced. For example:

# Input:
# ([{}])
# Output:
# True


class BalancedParenthesis:

    def is_balanced(self,str):
        pCnt = 0
        fCnt = 0
        sCnt = 0
        stack = []
        for x in str:
            if x in ('(','{','['):

                if x == '(':
                    if(pCnt<1):
                        pCnt +=1
                        stack.append(x)
                    else:
                        return False
                elif x == '{':
                    if(fCnt<1):
                        fCnt+=1
                        stack.append(x)
                    else:
                        return False
                elif x == '[':
                    if(sCnt<1):
                        sCnt+=1
                        stack.append(x)
                    else:
                        return False

                


            
            # clossing
            elif x == ')':
                if(pCnt == 0):
                    return False
                if stack[-1] == '(':
                    pCnt -=1
                    stack.pop()
                else:
                    return False
            elif x == ']':
                if(sCnt == 0):
                    return False
                if stack[-1] == '[':
                    sCnt -= 1
                    stack.pop()
                else:
                    return False
            elif x == '}':
                if(fCnt == 0):
                    return False
                if stack[-1] == '{':
                    fCnt -=1
                    stack.pop()
                else:
                    return False

        if(len(stack) == 0):
            return True
        else:
            return False

str = input("Enter the string : ")
obj = BalancedParenthesis();
print(obj.is_balanced(str))
