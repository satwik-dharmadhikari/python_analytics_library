from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Transformo las variables string a números

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # Arreglo de columnas de interés

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        '''
        Transfomra las columnas indicadas en el array columns,
        si no se especifica, transforma todas las columnas.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
