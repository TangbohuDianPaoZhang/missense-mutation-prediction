import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from torch import utils


def load_data(csv_name, root_dir, batch_size):
    csv_file = os.path.join(root_dir, csv_name)
    df = pd.read_csv(csv_file)
    feature_columns = ['MetaRNN_rankscore', 'MVP_rankscore', 'CADD_raw_rankscore_hg19',
                  'PrimateAI_rankscore', 'Polyphen2_HVAR_rankscore',
                  'kGp3_AF', 'ExAC_AF', 'gnomAD_exomes_AF']
    target_columns = 'True Label'

    feature_data = df[feature_columns]
    target_data = df[target_columns]
    input_dim = feature_data.shape[1]

    imputer = KNNImputer(n_neighbors=3)
    feature_data = pd.DataFrame(imputer.fit_transform(feature_data), columns=feature_columns)

    feature_tensor = torch.tensor(feature_data.values, dtype=torch.float32)
    target_tensor = torch.tensor(target_data.values, dtype=torch.int64)

    train_dataset = utils.data.TensorDataset(feature_tensor, target_tensor)
    train_dataloder = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return input_dim, train_dataloder
