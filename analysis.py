import pandas as pd

# Read the dataset and print metadata
column_names = ["polarity", "title", "text"]
df_train = pd.read_csv("train.csv", names=column_names, header=None)

df_train["polarity"] = df_train["polarity"].map(
    {1: 0, 2: 1}
)  # 0 for negative, 1 for positive

print(df_train.head())
print(df_train.info())
