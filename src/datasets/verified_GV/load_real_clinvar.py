import pandas as pd


def load_real_clinvar(csv_path='./root/data/verified_real_clinvar.csv'):
    records = []
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        record = {
            'Chromosome': row['Chr_hg38'],
            'Position': row['Start_hg38'],
            'Reference Base': row['Ref'],
            'Alternate Base': [row['Alt']],
            'ID': row['avsnp147'],
            'class': row['Class'],
            'sample': row['Sample ID'],
            'phenotype': row['HGMD_Phen'],
            'Func.refGene': row['Func.refGene'],
        }
        records.append(record)
    return records
