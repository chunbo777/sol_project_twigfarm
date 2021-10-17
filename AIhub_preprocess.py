import pandas as pd
from sklearn.model_selection import train_test_split
df_original = pd.read_csv("/home/tf-dev-01/workspace_sol/style-transfer/predata/pretest_train.0.csv")
df_translated = pd.read_csv("/home/tf-dev-01/workspace_sol/style-transfer/predata/pretest_train.1.csv")

# X_train, X_test, y_train, y_test = train_test_split(
# ...     X, y, test_size=0.33, random_state=42)

df_plus=pd.DataFrame()
df_plus["original"]=df_original
df_plus["tranlated"]=df_translated


X_train, X_test = train_test_split(df_plus, test_size=0.2, random_state=42)
X_train.to_csv("/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer/data/AIhub_train.csv")
X_test.to_csv("/home/tf-dev-01/workspace_sol/style-transfer/NLP_text-style-transfer/data/AIhub_test.csv")
print(df_plus)

