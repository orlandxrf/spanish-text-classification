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
duplicates = {}
categories = {}
total = 0
with open(filename, 'r') as f:
	for i, row in enumerate(f):
		if i==0: continue # skip headers
		# if i==100: break
		total += 1
		sys.stdout.write( '\tProcesando {} oraciones ...\r'.format( format(i, ',d') ) )
		sys.stdout.flush()
		category, id_doc, sentence = row.replace('\n','').split('\t')
		if sentence not in sentences:
			sentences[sentence] = (category, id_doc)
			if category not in categories:
				categories[category] = {'ok':1}
			else:
				categories[category]['ok'] += 1
		else:
			if category not in duplicates:
				duplicates[category] = {'dup':1}
			else:
				duplicates[category]['dup'] += 1
f.close()
print ('\n\tTerminado!\n')

categories = dict(sorted( categories.items(), key=lambda x:x[1]['ok'], reverse=True ))

print ( '\t#\torig\tdup\ttotal\tcategory' )
for i, cat in enumerate(categories):
	rest = categories[cat]['ok'] - duplicates[cat]['dup']
	print ( '\t{}\t{}\t{}\t{}\t{}'.format(i+1, format(categories[cat]['ok'], ',d'), format(duplicates[cat]['dup'], ',d'), format(rest, ',d'), cat) )
print ('\n\tTerminado!\n')

print ( '\t{}\tTotal de oraciones'.format( format(total,',d') ) )
print ( '\t{}\tOraciones a guardar'.format( format(len(sentences),',d') ) )
print ( '\t{}\tOraciones duplicadas\n'.format( format(total-len(sentences), ',d') ) )

for i, snt in enumerate(sentences):
	sys.stdout.write( '\tGuardando {} oraciones ...\r'.format( format(i+1, ',d') ) )
	sys.stdout.flush()
	data = '{}\t{}\t{}\n'.format(sentences[snt][0], sentences[snt][1], snt)
	saveIntoFile(fnsave, data)
print ('\n\tTerminado!\n')

