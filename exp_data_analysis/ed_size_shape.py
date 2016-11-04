
def show_info(df):
	print('Inspección de Datos **************************************************')
	print('Data Frame Train/Test')
	print(len(df))
	print(df[1:10])
	print(list(df.columns))

	print(df.describe())
