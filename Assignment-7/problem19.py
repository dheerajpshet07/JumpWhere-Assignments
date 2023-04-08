# 19. Write a Python program to remove duplicates from a list of lists.
# Sample list : [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
# New List : [[10, 20], [30, 56, 25], [33], [40]]

sample_list = [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]

new_list = list(set(tuple(sorted(sub)) for sub in sample_list))

print("After removing duplicates : ",new_list)