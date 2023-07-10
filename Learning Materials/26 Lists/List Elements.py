# Create a list with 4 elements of different datatypes
list_1 = [1, 1.1, "one", True]

# Create a list with 10 elements equal to 0
list_2 = [0]*10

# Nest list_1 and list_2 into a new list called list_3
list_3 = [list_1, list_2]

# Access the 4th element of list_1 and list_2 and assign them to list_4
list_4 = [list_1[3], list_2[3]]
print(list_3)