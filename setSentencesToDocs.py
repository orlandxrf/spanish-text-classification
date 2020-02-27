import sys

def saveIntoFile(fname, data, mode='a'):
	"""Save your data into file"""
	g = open(fname, mode)
	g.write(data)
	g.close()

filename = 'data/sentences_ok.txt'
fnsave = 'data/corpora_ok.txt'
data = 'id_doc\tdoc\n'
saveIntoFile(fnsave, data, 'w')

tmp_id = 0
document = []
with open(filename, 'r') as f:
	for i, row in enumerate(f):
		if i==0: continue # skip headers
		sys.stdout.write( '\tProcesando {} oraciones...\r'.format( format(i,',d') ) )
		sys.stdout.flush()
		catgeory, id_doc, sentence = row.replace('\n','').split('\t')
		if tmp_id == 0: tmp_id = id_doc
		if id_doc != tmp_id:
			data = '{}\t{}\n'.format(tmp_id, ' '.join(document))
			saveIntoFile(fnsave, data)
			tmp_id = id_doc
			document = []
		else: document += sentence.split()
f.close()
data = '{}\t{}\n'.format(tmp_id, ' '.join(document))
saveIntoFile(fnsave, data)
print ('\n\tTerminado!\n')
