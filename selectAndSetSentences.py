import sys

def saveIntoFile(fname, data, mode='a'):
	"""Save your data into file"""
	g = open(fname, mode)
	g.write(data)
	g.close()

filename = 'data/sentences_corpora.txt'
fnsave = 'data/sentences_ok.txt'
data = 'category\tid_doc\tsentence\n'
saveIntoFile(fnsave, data, 'w')

sentences = {}
categories = {}
with open(filename, 'r') as f:
	for i, row in enumerate(f):
		if i==0: continue # skip headers
		sys.stdout.write( '\tProcesando {} oraciones ...\r'.format( format(i), ',d') )
		sys.stdout.flush()
		category, id_doc, sentence = row.replace('\n','').split('\t')
		if sentence not in sentences:
			sentences[sentence] = 1
			if category not in categories:
				categories[category] = 1
			else:
				categories[category] += 1
f.close()


