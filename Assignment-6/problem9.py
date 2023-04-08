
# 9. A shop will give discount of 10% if the cost of purchased quantity is more than 1000.
# Ask user for quantity
# Suppose, one unit will cost 100.
# Judge and print total cost for user.

def calculateTotalCost(unit,pricePerUnit):
    totalCost = unit * pricePerUnit
    if(totalCost > 1000):
        totalCost = totalCost - (totalCost/10);
    return totalCost

qty = int(input("Enter the quantity:"))
print("Total cost is : ",calculateTotalCost(qty,100) )

