# Para manipular archivos grandes

import contextlib


chunksize = 50*10**4
fid = 1
with open('IP_BASE_PORT_IN_TRAIN_F.txt', 
	encoding='cp1252') as infile:
    f = open('file%d.txt' %fid, 'w')
    for i,line in enumerate(infile):
        f.write(line)
        if not i%chunksize:
            f.close()
            fid += 1
            f = open('file%d.txt' %fid, 'w')
    f.close()