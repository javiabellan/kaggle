import pandas as pd
from sklearn import model_selection


df = pd.read_csv("../data/labeledTrainData.tsv", sep='\t')
df["Fold"] = -1

y = df.sentiment.values
skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
	df.loc[val_idx, "Fold"] = fold

df.to_csv("../data/trainData_withFolds.csv", index=False)