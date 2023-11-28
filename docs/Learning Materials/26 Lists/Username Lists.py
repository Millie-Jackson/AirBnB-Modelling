import random
import string

usernames = []

for i in range(10):
    length = random.randint(5, 10)
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    usernames.append(username)

print(usernames)

# Print the datatype of usernames
print(type(usernames))

# Print the length of usernames
print(len(usernames))

# Print the datatype of the first element of usernames
print(type(usernames[0]))

# Make a new list called usernames_2 containing the last 5 elements of
usernames_2 = usernames[-5:]

# Remove the second element of usernames_2 and assign it to a variable called user_example
user_example = usernames_2.pop(1)
print(user_example)

# Using sort(), sort the elements of usernames alphabetically
usernames.sort()
print(usernames)

# Find the index of user_example in usernames and assign it to idx_user_example
idx_user_example = usernames.index(user_example)
# Retrieve the corresponding element in usernames, convert it to uppercase and assign it to upper_user_example
upper_user_example = usernames[idx_user_example].upper()
# Replace the corrisponding element in usernames with upper_user_example
usernames[idx_user_example] = upper_user_example
