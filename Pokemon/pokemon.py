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

''' MODIFYING DATA

# Adding up all numerical stats into a new Total column
# 45+49+49+65+65+45 = 318
df["Total"] = df["HP"] + df["Attack"] + df["Defense"] + df["Sp. Atk"] + df["Sp. Def"] + df["Speed"]
df["Total"] = df.iloc[:, 4:10].sum(axis=1) # Same thing
print(df.head(5))

# Removes total column
df = df.drop(columns=["Total"])
print(df.head(5))

# Reorder columns
df = df[["Total", "HP", "Defense"]]
cols = list(df.columns.values)
df = df[cols[0:4] + [cols[-1]] + cols[4:12]] # Same thing

'''


# and = & or = 
# Assign to new dataframe
new_df = df.loc[(df["Type 1"] == "Grass") | 
                (df["Type 2"] == "Poison") &
                (df["HP"] > 70)]
print(new_df.head(3))

# Old index will still be there 
# Re index and remove old index 
new_df = new_df.reset_index(drop=True)

# Save changed data to a new .csv
df.to_csv("AirBnB/Pokemon/modified.csv", index=False) # Removing indexs is optional
new_df.to_csv("AirBnB/Pokemon/filtered.csv", index=False)