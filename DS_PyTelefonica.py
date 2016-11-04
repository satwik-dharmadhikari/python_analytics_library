# *****************************************************************************
# Proyecto: Portabilidad IN
#
# Descripción:
#
# run with $time python Port_IN.py
#
# Task
# * Usar OOP (Oriented Object Programming)
# *****************************************************************************

import numpy as np
import pandas as pd
import pdb # Debugger
from sys import exit

# Telefonica Library
import get_and_clean.gc_miss as gc
import get_and_clean.gc_label_encoder as le
import exp_data_analysis.ed_size_shape as sh

# Stage 1: Getting and Cleaning Data
# ==================================

df = pd.read_table('IP_BASE_PORT_IN_TRAIN_F.txt', sep=";",
    encoding='cp1252',
    dtype={'id_phone': str, # In Train
    'Operator_Cd': np.int64, # In Train
    'operador': str, # In Train
    'Q_ANIS_SNA': np.float64, # In Train
    'WEIGHT': np.float64, # In Train
    'DIR': np.float64, # In Train
    'Q_CONTAC_POS': np.float64, # In Train
    'Q_CONTAC_PRE': np.float64, # In Train
    'Q_DIR_MAYOR_05': np.float64, # In Train
    'Q_DIR_MENOR_05': np.float64, # In Train
    'WEIGHT_5': np.float64, # In Train
    'DIR_5': np.float64, # In Train
    'Q_CONTAC_POS_5': np.float64, # In Train
    'Q_CONTAC_PRE_5': np.float64, # In Train
    'Q_DIR_MAYOR_05_5': np.float64, # In Train
    'Q_DIR_MENOR_05_5': np.float64, # In Train
    'PESO_1': np.float64, # In Train
    'DIR_1': np.float64, # In Train
    'access_destino': str, # In Train
    'FL_COM_ON_NET': np.float64, # In Train
    'MIEMBROS_COM': np.float64, # In Train
    'MIEMBROS_COM_MVS': np.float64, # In Train
    'MIEMBROS_COM_CLARO': np.float64, # In Train
    'MIEMBROS_PERS': np.float64, # In Train
    'FL_ON_NET_5': np.float64, # In Train
    'FL_COM_MAYOR_5': np.float64, # In Train
    'AVG_ANT_MESES': np.float64, # In Train
    'SUM_REC': np.float64, # In Train
    'SUM_FACT_N2': np.float64, # In Train
    'Q_RED_LTE': np.float64, # In Train
    'Q_RED_3G': np.float64, # In Train
    'Q_RED_2G': np.float64, # In Train
    'Q_TRAF_LTE': np.float64, # In Train
    'TIPO_EVENTO': str, # In Train
    'AGENTE': str,
    'MARCA_ES_EMPRESA': str,
    'MARCA_FUE_MOVISTAR': str,
    'CALIFICA_UNIVERSO': str,
    'RANGO_ARPU': str,
    'OFERTA': str,
    'PRECIO_PORTA': np.float64,
    'PRODUCTO': str,
    'cd_interurbano': np.float64,
    'MARCA_PORTO': str,
    'Q_MESES_CAMP': np.float64,
    'cd_interurbano_F': np.int64,
    'fl_delivery': np.int64,
    'Q_SIN_LLAMAR': np.int64,
    'Q_YA_POSEE_PTA': np.int64,
    'Q_NO_ACEPTA': np.int64,
    'Q_SIN_CONTACTO': np.int64,
    'Q_NO_ES_TIT': np.int64,
    'Q_ACEPTA': np.int64,
    'Q_DNI_INEXISTENTE': np.int64,
    'Q_RECHAZADO': np.int64,
    'Q_REGENDADO': np.int64,
    'Q_S_CONTACTOS_AG': np.int64,
    'Q_X_CONTACTOS_AG': np.int64,
    'Q_N_CONTACTOS_AG': np.int64,
    'target_julio': np.int64,
    'target_agosto': np.int64,
    'TARGET_F': np.int64,
    'Q_LLAM_ENT_N1': np.int64, #In Completa
    'SUM_SEG_ENT_N1': np.int64,
    'Q_ANIS_DIST_ENT_N1': np.int64,
    'Q_LLAM_SAL_N1': np.int64,
    'SUM_SEG_SAL_N1': np.int64,
    'Q_ANIS_DIST_SAL_N1': np.int64,
    'Q_LLAM_ENT_N2': np.int64,
    'SUM_SEG_ENT_N2': np.int64,
    'Q_ANIS_DIST_ENT_N2': np.int64,
    'Q_LLAM_SAL_N2': np.int64,
    'SUM_SEG_SAL_N2': np.int64,
    'Q_ANIS_DIST_SAL_N2': np.int64,
    'Q_LLAM_ENT_N3':np.int64,
    'SUM_SEG_ENT_N3': np.int64,
    'Q_LLAM_SAL_N3': np.int64,
    'SUM_SEG_SAL_N3': np.int64,
    'Q_ANIS_DIST_SAL_N3': np.int64,
    'LOC_VIVE': str,
    'CELDA_VIVE': np.float64,
    'CELDA_TRABAJA': np.float64,
    'LOC_TRABAJA': str,
    'SITIO_VIVE': str,
    'SITIO_TRABAJA': str,
    'SECTOR_VIVE': str,
    'SECTOR_TRABAJA': str,
    'MPD_VIVE': np.float64,
    'MPD_TRABAJA': np.float64,
    'q_llamadas_call_com': np.int64,
    'q_cont_c_reclamos': np.int64})

# Step 2: View the column names / summary of the dataset
# ******************************************************

# Inspección de datos

    
# Nota: La variable 'Precios Porta' se ve sospechosa, ya que no contiene valores. Revisamos esto

# print(df['PRECIO_PORTA'].count())

# Sept 3: Identify the variables with missing values
# **************************************************
    
# Cuento el numero de filas con missing

print(gc.miss_val_rows(df))

# Ahora analizo cuantos NaN hay por columnas, y su relación porcentual

print(gc.missing_values_table(df))

# Imprimo los valores con mayor cantidad de Missing
# 
# All NaN -> 0
# 
# Columnas                      Total NaN         % NaN
# FL_COM_ON_NET                134020            53.6080
# MIEMBROS_COM                 134020            53.6080
# MIEMBROS_COM_MVS             134020            53.6080
# MIEMBROS_COM_CLARO           134020            53.6080
# MIEMBROS_PERS                134020            53.6080
# FL_ON_NET_5                  134020            53.6080
# FL_COM_MAYOR_5               134020            53.6080
# TIPO_EVENTO                  229017            91.6068 NaN -> NO1
# AGENTE                       217445            86.9780 NaN -> NO2
# MARCA_ES_EMPRESA             217445            86.9780 NaN -> No
# MARCA_FUE_MOVISTAR           217518            87.0072 NaN -> No
# CALIFICA_UNIVERSO            217933            87.1732 NaN -> No
# RANGO_ARPU                   218632            87.4528 NaN -> Sacamos
# OFERTA                       219286            87.7144 NaN -> 0: binaria -> Tiene 14 categorias
# PRECIO_PORTA                 250000           100.0000 Sacar
# PRODUCTO                     217445            86.9780 NaN -> 0: binaria -> Tiene 6 categorias
# cd_interurbano               219101            87.6404 Sacamos
# MARCA_PORTO                  217445            86.9780 NaN -> 0: binaria -> Corregida a Binaria
# Q_MESES_CAMP                 116296            46.5184 NaN -> No: categorica
# 
# Dropeo columnas que no son de interés
# SECTOR_VIVE = CELDA_VIVE
# SECTOR_TRABAJA = CELDA_TRABAJA

df.is_copy = False # Elimina un warning sobre copia de df, no altera los valores

df.drop(['id_phone', 
    'operador', 
    'target_agosto', 
    'TARGET_F', 
    'PRECIO_PORTA',
    'RANGO_ARPU',
    'cd_interurbano',
    'access_destino',
    'AGENTE',
    'MARCA_ES_EMPRESA',
    'MARCA_FUE_MOVISTAR',
    'CALIFICA_UNIVERSO',
    'OFERTA',
    'PRECIO_PORTA',
    'PRODUCTO',
    'cd_interurbano',
    'MARCA_PORTO',
    'Q_MESES_CAMP',
    'cd_interurbano_F',
    'fl_delivery',
    'Q_SIN_LLAMAR',
    'Q_YA_POSEE_PTA',
    'Q_NO_ACEPTA',
    'Q_SIN_CONTACTO',
    'Q_NO_ES_TIT',
    'Q_ACEPTA',
    'Q_DNI_INEXISTENTE',
    'Q_RECHAZADO',
    'Q_REGENDADO',
    'Q_S_CONTACTOS_AG',
    'Q_X_CONTACTOS_AG',
    'Q_N_CONTACTOS_AG',
    'SECTOR_VIVE',
    'SECTOR_TRABAJA',
    'SITIO_VIVE',
    'SITIO_TRABAJA',
    'LOC_VIVE',
    'LOC_TRABAJA',
    'q_llamadas_call_com',
    'q_cont_c_reclamos'
    ], axis = 1, inplace = True)

# Elimino por no poder tratar las variables, una idea puede ser convertir a categoria 
# numerica a mano, es decir: Ordenar alfabeticamente y luego asignar un número.

# ## Convierto NaN a categorias

# Analizo la variable TIPO_EVENTO

#print('describe TIPO_EVENTO')

#df['TIPO_EVENTO'].describe()


df['TIPO_EVENTO'].fillna('NO1', inplace=True)
#df['AGENTE'].fillna('NO2', inplace=True)
#df['MARCA_ES_EMPRESA'].fillna('NO3', inplace=True)
#df['MARCA_FUE_MOVISTAR'].fillna('NO4', inplace=True)
#df['CALIFICA_UNIVERSO'].fillna('NO5', inplace=True)
#df['OFERTA'].fillna('NO6', inplace=True)
#df['PRODUCTO'].fillna('NO7', inplace=True)
#df['MARCA_PORTO'].fillna('NOPORTO', inplace=True)
#df['Q_MESES_CAMP'].fillna(12, inplace=True)

# Binarizo Variables
# ******************

#pdb.set_trace()

print('Info df, para ver numero de variables')

print(sh.show_info(df))

exit(0)

df2 = le.MultiColumnLabelEncoder(columns = ['TIPO_EVENTO']).fit_transform(df)

# Tratamiento de los NaN

gc.missing_values_table(df2)

# Step 2: Exploratory Data Analysis
# =================================

print('*******************Correlation***********************')

print(df2.corr()['target_julio'])


# Step 3: Construcción del Modelo
# ===============================

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth = 30, min_samples_split=2, 
    n_estimators = 400, random_state = 1, n_jobs=3, verbose=1)

import time

start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))
# Pre process
# ***********

#from sklearn.preprocessing import Imputer

#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

print('**************************************************')

#df2 =imp.fit_transform(df2)

df3 = df2.fillna(df2.mean())

print(gc.missing_values_table(df3))


# Defino las variables Target y de Entrenamiento
# **********************************************

from sklearn.model_selection import train_test_split

trainprepare, test = train_test_split(df3, test_size = 0.3)

target = trainprepare["target_julio"]

train = trainprepare.drop("target_julio", axis=1) 

# Training the model
# ******************

print("--- %s seconds ---" % (time.time() - start_time))

model.fit(train, target)

print("Train --- %s seconds ---" % (time.time() - start_time))

# Features Importance
# *******************

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

X = train
y = target

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking

# Agregar nombres a las columnas para hacer el print

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]), indices)
#plt.xlim([-1, X.shape[1]])
#plt.show()

# testing the model
# *****************

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

predictions = model.predict(test.drop("target_julio",axis=1))

print(predictions)

print('Print roc_auc_score')

print(roc_auc_score(predictions, test["target_julio"]))

#print('Cross_val_score')

#print(cross_val_score(model, train, target).mean())

# Save trained model
# ******************

print("Pickle--- %s seconds ---" % (time.time() - start_time))

import pickle

# save the model to disk

filename = 'trained_port_in.sav'

pickle.dump(model, open(filename, 'wb'))

#print('load the model from disk')

#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(test.drop("target_julio",axis=1),test["target_julio"])

#print(result)

print("End Pickle--- %s seconds ---" % (time.time() - start_time))

# Get the score and add to test
# *****************************

final_df = model.predict_proba(test.drop("target_julio",axis=1))[:,1]

test['score'] = final_df

print(test[['target_julio', 'score']].sort(['score'], ascending=[False]).head(100))

# =====================================================================================
# Ahora tengo que preparar el dataset para aplicar el modelo
# Al caso real
# =====================================================================================

#Transformo las variables

df_lastest = pd.read_table('IP_BASE_PORT_IN_COMPLETA.txt', sep=";", nrows=2000,
    encoding='cp1252',
    dtype={'id_phone': str, # In Train
    'Operator_Cd': np.int64, # In Train
    'operador': str, # In Train
    'Q_ANIS_SNA': np.float64, # In Train
    'WEIGHT': np.float64, # In Train
    'DIR': np.float64, # In Train
    'Q_CONTAC_POS': np.float64, # In Train
    'Q_CONTAC_PRE': np.float64, # In Train
    'Q_DIR_MAYOR_05': np.float64, # In Train
    'Q_DIR_MENOR_05': np.float64, # In Train
    'WEIGHT_5': np.float64, # In Train
    'DIR_5': np.float64, # In Train
    'Q_CONTAC_POS_5': np.float64, # In Train
    'Q_CONTAC_PRE_5': np.float64, # In Train
    'Q_DIR_MAYOR_05_5': np.float64, # In Train
    'Q_DIR_MENOR_05_5': np.float64, # In Train
    'PESO_1': np.float64, # In Train
    'DIR_1': np.float64, # In Train
    'access_destino': str, # In Train
    'FL_COM_ON_NET': np.float64, # In Train
    'MIEMBROS_COM': np.float64, # In Train
    'MIEMBROS_COM_MVS': np.float64, # In Train
    'MIEMBROS_COM_CLARO': np.float64, # In Train
    'MIEMBROS_PERS': np.float64, # In Train
    'FL_ON_NET_5': np.float64, # In Train
    'FL_COM_MAYOR_5': np.float64, # In Train
    'AVG_ANT_MESES': np.float64, # In Train
    'SUM_REC': np.float64, # In Train
    'SUM_FACT_N2': np.float64, # In Train
    'Q_RED_LTE': np.float64, # In Train
    'Q_RED_3G': np.float64, # In Train
    'Q_RED_2G': np.float64, # In Train
    'Q_TRAF_LTE': np.float64, # In Train
    'TIPO_EVENTO': str, # In Train
    # AGENTE
    # MARCA_ES_EMPRESA
    # MARCA_FUE_MOVISTAR
    # RANGO_ARPU
    # CALIFICA_UNIVERSO
    # OFERTA
    # PRECIO_PORTA
    # PRODUCTO
    # cd_interurbano
    # MARCA_PORTO
    # Q_MESES_CAMP
    # cd_interurbano_F
    # fl_delivery
    # Q_SIN_LLAMAR
    # Q_YA_POSEE_PTA
    # Q_NO_ACEPTA
    # Q_SIN_CONTACTO
    # Q_NO_ES_TIT
    # Q_ACEPTA
    # Q_DNI_INEXISTENTE
    # Q_RECHAZADO
    # Q_REGENDADO
    # Q_S_CONTACTOS_AG
    # Q_X_CONTACTOS_AG
    # Q_N_CONTACTOS_AG
    # target_julio
    # target_agosto
    # TARGET_F
    'target_JULIO': np.int64, # Cambiar Nombre
    'target_AGOSTO': np.int64, # Cambiar Nombre
    'Q_LLAM_ENT_N1': np.float64, # In Train
    'SUM_SEG_ENT_N1': np.float64, # In Train
    'Q_ANIS_DIST_ENT_N1': np.float64, # In Train
    'Q_LLAM_SAL_N1': np.float64, # In Train
    'SUM_SEG_SAL_N1': np.float64, # In Train
    'Q_ANIS_DIST_SAL_N1': np.float64, # In Train
    'Q_LLAM_ENT_N2': np.float64, # In Train
    'SUM_SEG_ENT_N2': np.float64, # In Train
    'Q_ANIS_DIST_ENT_N2': np.float64, # In Train
    'Q_LLAM_SAL_N2': np.float64, # In Train
    'SUM_SEG_SAL_N2': np.float64, # In Train
    'Q_ANIS_DIST_SAL_N2': np.float64, # In Train
    'Q_LLAM_ENT_N3': np.float64, # In Train
    'SUM_SEG_ENT_N3': np.float64, # In Train
    'Q_LLAM_SAL_N3': np.float64, # In Train
    'SUM_SEG_SAL_N3': np.float64, # In Train
    'Q_ANIS_DIST_SAL_N3': np.float64, # In Train
    'LOC_VIVE': str, # In Train
    'CELDA_VIVE': np.float64, # In Train
    'CELDA_TRABAJA': np.float64, # In Train
    'LOC_TRABAJA': str, # In Train
    'SITIO_VIVE': str, # In Train
    'SITIO_TRABAJA': str, # In Train
    'SECTOR_VIVE': str, # In Train
    'SECTOR_TRABAJA': str, # In Train
    'MPD_VIVE': np.float64, # In Train
    'MPD_TRABAJA': np.float64}) # In Train
    # q_llamadas_call_com
    # q_cont_c_reclamos 

df_lastest.drop(['id_phone',
    'operador',
    'target_AGOSTO',
    'target_JULIO',
    'access_destino',
    'SECTOR_VIVE',
    'SECTOR_TRABAJA',
    'SITIO_VIVE',
    'SITIO_TRABAJA',
    'LOC_VIVE',
    'LOC_TRABAJA'
    ], axis = 1, inplace = True)

df_lastest.is_copy = False # Elimina un warning sobre copia de df, no altera los valores

# Esto debo convertir en method, para reutilizar

df_lastest['TIPO_EVENTO'].fillna('NO1', inplace=True)
#df_lastest['AGENTE'].fillna('NO2', inplace=True)
#df_lastest['MARCA_ES_EMPRESA'].fillna('NO3', inplace=True)
#df_lastest['MARCA_FUE_MOVISTAR'].fillna('NO4', inplace=True)
#df_lastest['CALIFICA_UNIVERSO'].fillna('NO5', inplace=True)
#df_lastest['OFERTA'].fillna('NO6', inplace=True)
#df_lastest['PRODUCTO'].fillna('NO7', inplace=True)
#df_lastest['MARCA_PORTO'].fillna('NOPORTO', inplace=True)
#df_lastest['Q_MESES_CAMP'].fillna(12, inplace=True)

print('Describe df_lastest')

df_lastest_f = df_lastest.fillna(df_lastest.mean())
print(df_lastest_f.describe())
print('Info df_lastest')
print(df_lastest_f.describe())

#df_lastest2 = MultiColumnLabelEncoder(columns = ['TIPO_EVENTO','AGENTE','MARCA_ES_EMPRESA',
#    'MARCA_FUE_MOVISTAR','CALIFICA_UNIVERSO','OFERTA','PRODUCTO','MARCA_PORTO']).fit_transform(df_lastest_f)

df_lastest2 = MultiColumnLabelEncoder(columns = ['TIPO_EVENTO']).fit_transform(df_lastest_f)

print('Flag 1')
#final_df_lastest2 = model.predict_proba(df_lastest2.drop("target_julio",axis=1))[:,1]

final_df_lastest2 = model.predict_proba(df_lastest2)[:,1]
print('Flag 2')
df_lastest['score'] = final_df_lastest2

print('Final Table Scored')
#print(df_original[0:20])
#print(df_original[['id_phone', 'target_JULIO', 'score']])
#print(len(df_original))
print(df_lastest[0:20])