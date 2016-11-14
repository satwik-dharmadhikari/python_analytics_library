# Para manipular archivos grandes

#import logging.handlers

#log = logging.getLogger()
#fh = logging.handlers.RotatingFileHandler("TRAIN_Frac.txt", 
#     maxBytes=2**20*100, backupCount=100, encoding='cp1252') 
# 100 MB each, up to a maximum of 100 files
#log.addHandler(fh)
#log.setLevel(logging.INFO)
#f = open("IP_BASE_PORT_IN_TRAIN_F.txt")

#while True:
	#log.info(f.readline().strip())

import os
import subprocess


def split_file(file, size):

	msn = 'split --bytes '+str(size)+'M --numeric-suffixes --suffix-length=2 '+ file+' '+file+'par'

	print(msn)
	
	print('Fraccionando archivos...')
	
	os.system(msn)

# Tomo la primera linea del archivo f_get_header y agrego a f_add_header

def split_add_header(f_get_header, f_add_header):

	msn1 = 'head -1 ' + f_get_header +' | tail -1'

	header = os.system(str(msn1))

	print('******************header***********************')

	# Warning Arreglar, no guarda bien el mensaje en header
	# Buscar el modo de guardar adecuadamente el header

	header = str(subprocess.check_output(msn1, shell=True))

	print("program output:" + str(header))
	
	exit(0)

	sed_sentence = 'sed -i 1i'+ str(header) +' '+ f_add_header

	print('Agregando encabezados...')
	
	os.system(sed_sentence)

	print('Verificando los encabezados nuevos...')

	msn2 = 'head -1 ' + f_add_header +' | tail -1'

	print(os.system(msn2))
	print('***************END PROCESS****************')
