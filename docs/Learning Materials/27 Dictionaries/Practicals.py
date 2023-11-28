my_dict = {"Key_1": 1, "Key_2": 2}

print(my_dict["Key_1"])
print(my_dict["Key_2"])
#print(my_dict["Key_3"])
print(my_dict.keys())
print(my_dict.values())
print(my_dict.values())

print(my_dict.items())
my_items = my_dict.items()
print(type(my_items))

my_dict["Key_3"] = 3
new_key = {"Key_4": 4}
my_dict.update(new_key)
print(my_dict)

del my_dict["Key_1"]
print(my_dict)
my_value = my_dict.pop("Key_2")
print(my_value)