tomatoes = {"price": 0.87, "quantity": 6}
sponges = {"price": 0.29, "quantity": 10}
juice = {"price": 1.89, "quantity": 1500}
foil = {"price": 1.29, "quantity": 30}
sugar = {"price": 1.09, "quantity": 500}
'''
quantity = 2
sponge_cost = sponges["price"] * quantity
juice_cost = juice["price"] * quantity
tomato_cost = tomatoes["price"] * quantity
cost = sponge_cost + juice_cost + tomato_cost

print("1kg of sugar costs: £", sugar["price"] * 2)
print("20 sponges, 3l of juice and 2 packs of tomatoes costs: £", cost)
0.87
0.09
1.26
0.86
0.39

'''
customers = 5
tomato_cost = tomatoes["price"]
sponge_cost = sponges["price"] / sponges["quantity"] * 3
juice_cost = juice["price"] / juice["quantity"] * 1000
foil_cost = foil["price"] / foil["quantity"] * 20
sugar_cost = sugar["price"] / sugar["quantity"] * 180
total = (tomato_cost + sponge_cost + juice_cost + foil_cost + sugar_cost) * customers
print("Total for 5 customers :£", total)
vat = total / 20
print("Plus VAT :£", round(total + vat, 2))