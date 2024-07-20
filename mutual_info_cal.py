import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('dataset/traindata.csv')
column_to_drop = ['True Label', 'CHROM', 'Nuc-Pos', 'REF-Nuc', 'ALT-Nuc',
                  'Ensembl-Gene-ID', 'Ensembl-Protein-ID', 'Ensembl-Transcript-ID',
                  'Uniprot-Accession', 'Amino_acids', 'cDNA_position', 'CDS_position',
                  'Protein_position', 'Codons', 'DOMAINS', 'Consequence', 'IMPACT',
                  'kGp3_AF', 'ExAC_AF', 'gnomAD_exomes_AF', 'ClinVar_preferred_disease_name_in_CLNDISDB']
x = df.drop(column_to_drop, axis=1)
y = df['True Label']

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)
x = pd.get_dummies(x)

mi = mutual_info_classif(x, y)

mi_df = pd.DataFrame({'Feature': x.columns, 'MI': mi})
mi_df = mi_df.sort_values(by='MI', ascending=False)

print(mi_df.head(10))
