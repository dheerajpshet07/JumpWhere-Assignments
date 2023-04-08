# 10. A company decided to give bonus of 5% to employee if his/her year of service is more than 5 years.
# Ask user for their salary and year of service and print the net bonus amount.


def calcNetBonus(salary,yoe):
    if(yoe<=5):
        return 0
    else:
        return (salary*5)/100

salary = int(input("Enter the salary:"))
yoe = int(input("Enter the year of experience:"))

print("Net Bonus : ",calcNetBonus(salary,yoe))

