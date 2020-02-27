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
totalCategories = {}
originalCategories = {}
total = 0
with open(filename, 'r') as f:
	for i, row in enumerate(f):
		if i==0: continue # skip headers
		# if i==100: break
		total += 1
		sys.stdout.write( '\tProcesando {} oraciones ...\r'.format( format(i, ',d') ) )
		sys.stdout.flush()
		category, id_doc, sentence = row.replace('\n','').split('\t')
		# --------------------------------------------------------------
		# count number of original sentences by category
		if category not in originalCategories: originalCategories[category] = 1
		else: originalCategories[category] += 1
		# --------------------------------------------------------------
		if sentence not in sentences:
			sentences[sentence] = (category, id_doc)
			if category not in totalCategories: totalCategories[category] = 1
			else: totalCategories[category] += 1
f.close()
print ('\n\tTerminado!\n')

totalCategories = dict(sorted( totalCategories.items(), key=lambda x:x[1], reverse=True ))

print ( '\t#\torig\tdup\ttotal\tcategory' )
for i, cat in enumerate(totalCategories):
	rest =  originalCategories[cat] - totalCategories[cat]
	print ( '\t{}\t{}\t{}\t{}\t{}'.format(i+1, format(originalCategories[cat], ',d'), format(rest, ',d'), format(totalCategories[cat], ',d'), cat) )
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

