import pandas as pd

df = pd.read_csv("Disease and symptoms dataset.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)