# 15. Take integer inputs from user until he/she presses q ( Ask to press q to quit after every integer input ). Print average and product of all numbers.



def solution(nums):
    prod = 1
    sum = 0
    for x in nums:
        sum += x
        prod *= x
    avg = sum/len(nums)

    print("Avg : ",avg,"\nProduct : ",prod)

print("Enter integer number. \n Press q to stop")
nums = []
while(True):
    x = input()
    if(x=="q"):
        break
    nums.append(int(x))
solution(nums)
