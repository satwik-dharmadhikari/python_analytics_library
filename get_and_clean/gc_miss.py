import pandas as pd


def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return print(mis_val_table_ren_columns.sort_values('% of Total Values', ascending=False))

def miss_val_rows(df):
		sum(map(any, df.isnull()))