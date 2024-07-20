import pandas as pd
from sklearn.impute import KNNImputer

data = pd.read_csv('dataset/train_dataset.csv')
column_to_keep = ['MetaRNN_rankscore', 'MVP_rankscore', 'CADD_raw_rankscore_hg19',
                  'PrimateAI_rankscore', 'Polyphen2_HVAR_rankscore',
                  'kGp3_AF', 'ExAC_AF', 'gnomAD_exomes_AF', 'True Label']
df = data[column_to_keep]

imputer = KNNImputer(n_neighbors=3)
filled_df = pd.DataFrame(imputer.fit_transform(df), columns=column_to_keep)
filled_df.to_csv('dataset/train_dataset_filled(n=3).csv', index=False)
