
import pandas as pd

def show_info(df):
	print('Inspección de Datos **************************************************')
	print('Número de Registros en la tabla')
	print(len(df))
	print('Número de Features')
	print(df.shape)
	print('Info')
	print(df.info())
	print('Describe')
	print(df.describe())
