import sys

def saveToFile(fname, data, mode='a'):
	g = open(fname, mode)
	g.write(data)
	g.close()

def writeDocuments(count, filename, category, path_save):
	vocabulary = {}
	with open(filename, 'r') as f:
		for i, row in enumerate(f):
			if i==0: continue # skip headers
			sys.stdout.write('\t{} - procesando {} documentos ...\r'.format(category, format(i, ',d') ))
			sys.stdout.flush()
			__, doc = row.replace('\n','').split('\t')
			data = '{}\t{}\n'.format(category, doc)
			saveToFile(path_save, data)
			count += 1
	f.close()
	print ('\n\tTerminado!\n')
	return count



categories = {
	'crime': {'docs':0, 'vocabulary':0, },
	'culture': {'docs':0, 'vocabulary':0, },
	'economy': {'docs':0, 'vocabulary':0, },
	'finance': {'docs':0, 'vocabulary':0, },
	'gossip': {'docs':0, 'vocabulary':0, },
	'health': {'docs':0, 'vocabulary':0, },
	'justice': {'docs':0, 'vocabulary':0, },
	'police': {'docs':0, 'vocabulary':0, },
	'politics': {'docs':0, 'vocabulary':0, },
	'security': {'docs':0, 'vocabulary':0, },
	'show': {'docs':0, 'vocabulary':0, },
	'society': {'docs':0, 'vocabulary':0, },
	'spectacle': {'docs':0, 'vocabulary':0, },
	'sports': {'docs':0, 'vocabulary':0, },
	'technology':{'docs':0, 'vocabulary':0, },
}

corpora_path = 'data/corpora.txt'
data = 'category\tdocument\n'
saveToFile(corpora_path, data, 'w')

count = 1
for category in categories:
	fname = 'data/{}.txt'.format( category )
	count = writeDocuments(count, fname, category, corpora_path)

print ( '\nTerminado!\n' )
print ( '{} Total de documentos\n'.format( format(count, ',d') ) )
