empty_tuple = ()
empty_tuple = tuple()

my_tuple1 = (1, 1.1, "one")
my_tuple2 = (2, 2.2, "two")
my_tuple3 = my_tuple1 + my_tuple2

vowels = ("a", "e", "i", "o", "u")
print(vowels.count(vowels))

my_tuple = (1, "f", [1, 2, 3], [4, 5], ("f", "d"), ("f", "d", "e"), [1, 2, 3], "a")
print(my_tuple.count([1, 2, 3]))
print(my_tuple.count(("f", "d")))

print(vowels.index("e"))
print(vowels.index("i"))
#print(vowels.find("b"))

letters_tuple = ("a", "e", "i", "o", "u", "g", "l", "m", "n", "p", "r", "s", "t", "i", "v", "w", "y")
#print(letters_tuple[4:8].index("i"))

my_tuple = (70, "AiCore", 10, "Programming", 70)
x, *y, z = my_tuple
print(x)
print(y)
print(z)

x, y, *z = my_tuple
print(x)
print(y)
print(z)