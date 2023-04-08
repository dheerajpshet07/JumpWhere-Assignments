# 1. Write a Python script to sort (ascending and descending) a dictionary by value.

my_dict = { 'num6': 6, 'num3': 3, 'num2': 2, 'num4': 4, 'num1': 1, 'num5': 5}

sortedDictAsc = sorted(my_dict.items(), key=lambda x:x[1])
converted_sortedDictAsc = dict(sortedDictAsc)
print("Sorted dict by val in Ascending order: ",converted_sortedDictAsc)

sortedDictDesc = sorted(my_dict.items(), key=lambda x:x[1],reverse=True)
converted_sortedDictDesc = dict(sortedDictDesc)
print("\nSorted dict by val in Descending order: ",converted_sortedDictDesc)
