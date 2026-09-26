
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATA_PATH)

# Drop identifier columns not useful for prediction
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])
    
# Clean strings
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    
# Impute missing values prior to export
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    if col != "ProdTaken":
        df[col] = df[col].fillna(df[col].median())
        
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    
# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Perform  Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
