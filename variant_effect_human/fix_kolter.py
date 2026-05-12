import pandas as pd
with open("variant_effect_human/human_sequences_wt/P53_CDS.txt", 'r') as file:
    ref = file.read()

df1 = pd.read_excel("/Users/emmy/Downloads/mmc3.xlsx")
df1.rename(columns={"RFS_H1299": "DMS"}, inplace=True)
df2 = pd.read_excel("/Users/emmy/Downloads/mmc5.xlsx")
df2.rename(columns={"RFS_HCT116": "DMS2"}, inplace=True)
df = pd.concat([df1,df2])

df_new = df[df.Sec_codon_num != df.Sec_codon_num]
df_new = df_new[(df_new.Mut_type == "AASub") | (df_new.Mut_type == "Sub")]
df_new = df_new[df_new.Backbone == "wt"]
df_new = df_new[~df_new["AA_change"].str.contains("*", regex=False)]

def process_mutation(row, ref_string):
    nn_start_0 = row["Position"]-1
    before_change, after_change = row["Seq_change"].split(">")
    len_of_change = len(after_change)
    assert ref_string[nn_start_0:nn_start_0+len_of_change] == before_change
    mutated_sequence = ref_string[:nn_start_0] + after_change + ref_string[nn_start_0+len_of_change:]
    assert mutated_sequence[nn_start_0:nn_start_0+len_of_change] == after_change
    assert len(mutated_sequence) == len(ref_string)
    codon_num = int(row["Codon_num"])
    aa_before, aa_after = row["AA_change"].split(">")
    id_column = f"P53_{aa_before}{codon_num}{aa_after}_{before_change}>{after_change}"

    return pd.Series([id_column, mutated_sequence, row["DMS"], row["DMS2"]])


# Apply the function to the dataframe
df_final = df_new.apply(
    process_mutation,
    axis=1,
    ref_string=ref
)
df_final.columns = ["id", "mutated_sequence_dna", "DMS", "DMS2"]
df1 = df_final[df_final.DMS == df_final.DMS]
df2 = df_final[df_final.DMS2 == df_final.DMS2]
df1.to_csv("variant_effect_human/kotler.csv", index=False)
df2.to_csv("variant_effect_human/kotler2.csv", index=False)
print(df_final.head())