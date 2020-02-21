import sys

def readDocuments(filename, category):
	vocabulary = {}
	with open(filename, 'r') as f:
		for i, row in enumerate(f):
			if i==0: continue # skip headers
			sys.stdout.write('\t{} - procesando {} documentos ...\r'.format(category, format(i, ',d') ))
			sys.stdout.flush()
			id_doc, doc = row.replace('\n','').split('\t')
			for word in doc.split():
				if word not in vocabulary: vocabulary[word] = 1
				else: vocabulary[word] += 1
	f.close()
	print ('\n\tTerminado!\n')
	return (vocabulary, i)



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

# for category in categories:
# 	fname = 'categories/{}.txt'.format( category )
# 	vocabulary, ndocs = readDocuments(fname, category)
# 	categories[category]['docs'] = ndocs
# 	categories[category]['vocabulary'] = len(vocabulary)

# print ('\nTerminado!\n')


# print (categories)

# print ('')




categories = {'crime': {'docs': 5411, 'vocabulary': 38528}, 'culture': {'docs': 25239, 'vocabulary': 169443}, 'economy': {'docs': 22349, 'vocabulary': 86160}, 'finance': {'docs': 10872, 'vocabulary': 73658}, 'gossip': {'docs': 5576, 'vocabulary': 72893}, 'health': {'docs': 13215, 'vocabulary': 90452}, 'justice': {'docs': 21322, 'vocabulary': 81981}, 'police': {'docs': 81489, 'vocabulary': 121272}, 'politics': {'docs': 36341, 'vocabulary': 112715}, 'security': {'docs': 30751, 'vocabulary': 83542}, 'show': {'docs': 16516, 'vocabulary': 93371}, 'society': {'docs': 27255, 'vocabulary': 116259}, 'spectacle': {'docs': 41417, 'vocabulary': 135490}, 'sports': {'docs': 100085, 'vocabulary': 192717}, 'technology': {'docs': 12336, 'vocabulary': 93750}}

# print ( list(categories.items()) )
categories = dict( sorted(categories.items(), key=lambda x:x[1]['docs'], reverse=True) )

print ('-'*50)
print ('#\tcategory\t# docs\t# vocabulary')
print ('-'*50)
for i, category in enumerate(categories):
	print ( '{}\t{}\t\t{}\t{}'.format(i+1, category, format(categories[category]['docs'], ',d'), format( categories[category]['vocabulary'], ',d') ) )
print ('-'*50)
