# 12. A student will not be allowed to sit in exam if his/her attendence is less than 75%.
# Take following input from user
# Number of classes held
# Number of classes attended.
# And print
# percentage of class attended
# Is student is allowed to sit in exam or not.
def displayEligibility(numClass,numClassAttended):
    percentage_attended = (numClassAttended*100)/numClass
    print("Attendence percentage : ",percentage_attended)
    if(percentage_attended < 75.00):
        print("Not eligible to sit in exam")
    else:
        print("Eligible to sit in exam")

numClass = int(input("Enter the total num of class:"))
numClassAttended = int(input("Number of class attended:"))
displayEligibility(numClass,numClassAttended)