import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

        self.feature_columns = ['MetaRNN_rankscore', 'MVP_rankscore', 'M_CAP_rankscore',
                                    'CADD_raw_rankscore', 'MetaSVM_rankscore', 'SIFT_converted_rankscore']
        self.target_columns = 'True Label'

    def __len__(self):
        return len(self.df.groupby('True Label'))

    def __getitem__(self, idx):
        start = idx
        end = start + self.T
        target_end = end + self.S

        _input = torch.tensor(self.df[self.feature_columns].iloc[start:end].values, dtype=torch.float32)
        target = torch.tensor(self.df[self.target_columns].iloc[end:target_end].values, dtype=torch.int64)

        _input = torch.tensor(self.transform.fit_transform(_input), dtype=torch.float32)

        return _input, target



training_length = 100
forecast_window = 10

dataset = MyDataset(csv_name='test_dataset.csv', root_dir='dataset', training_length=training_length, forecast_window=forecast_window)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for _input, target in dataloader:
    print("Input shape:", _input.shape)
    print("Target shape:", target.shape)
    break


def load_data(csv_name, root_dir):
    csv_file = os.path.join(root_dir, csv_name)
    df = pd.read_csv(csv_file)
    feature_columns = ['MetaRNN_rankscore', 'MVP_rankscore', 'CADD_raw_rankscore_hg19',
                  'PrimateAI_rankscore', 'Polyphen2_HVAR_rankscore',
                  'kGp3_AF', 'ExAC_AF', 'gnomAD_exomes_AF']
    target_columns = 'True Label'

    feature_data = df[feature_columns]
    target_data = df[target_columns]

    imputer = KNNImputer(n_neighbors=3)
    feature_data = pd.DataFrame(imputer.fit_transform(feature_data), columns=feature_columns)

    feature_tensor = torch.tensor(feature_data.values, dtype=torch.float32)
    target_tensor = torch.tensor(target_data.values, dtype=torch.int64)

    return feature_tensor, target_tensor