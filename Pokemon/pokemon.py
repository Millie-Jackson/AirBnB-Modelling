import pandas as pd

df = pd.read_csv("AirBnB/Pokemon/pokemon_data.csv")

''' ACCESS

# First 5
print(df.head(5))

# Statistics
print(df.describe())

# Columns
print(df.columns)

# First 5 columns
print(df["Name"][0:5])

# Multiple columns
print(df[["Name", "Type 1", "HP"]])

# Row
print(df.iloc[1:4])

# Specific element
print(df.iloc[2, 1])

# Find more specific info
print(df.loc[df["Type 1"] == "Grass"])

# Iteration
for index, row in df.iterrows():
    print(index, row)

# Sort descending
print(df.sort_values("Name", ascending=False))    

'''

# Adding up all numerical stats into a new Total column
df["Total"] = df["HP"] + df["Attack"] + df["Defense"] + df["Sp. Atk"] + df["Sp. Def"] + df["Speed"]
print(df.head(5))