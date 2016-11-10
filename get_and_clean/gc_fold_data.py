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

def split_file(file, size):
	msn = 'split --bytes '+str(size)+'M --numeric-suffixes --suffix-length=2 '+ file+' '+file+'par'
	print(msn)
	os.system(msn)
