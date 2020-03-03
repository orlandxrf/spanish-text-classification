import fasttext
import fasttext.util
import sys
from datetime import datetime

start_time = datetime.now()

def saveIntoFile(fname, data, mode='a'):
	"""Save your data into file"""
	g = open(fname, mode)
	g.write(data)
	g.close()

def getVocabulary(filename):
	categories = {}
	vocabulary = {}
	count = 1
	with open(filename, 'r') as f:
		for i, row in enumerate(f):
			if i==0: continue # skip headers
			sys.stdout.write( '\tProcesando {} documento...\r'.format( format(i,',d') ) )
			sys.stdout.flush()
			category, doc = row.replace('\n','').split('\t')
			if category not in categories: categories[category] = 1
			else: categories[category] += 1
			for word in doc.split():
				if word not in vocabulary:
					vocabulary[word] = count
					count += 1
	f.close()
	print ( '\n\tTerminado!\n' )
	return (categories, vocabulary)

filename = 'data/corpora.txt'
categories, vocabulary = getVocabulary(filename)
categories = dict( sorted(categories.items(), key=lambda x:x[1], reverse=True) )
print ( '\tInfomación.\n' )
for cat in categories:
	print ( '\t{}\t{}'.format( cat, format(categories[cat], ',d') ) )
print ('')
print ( '\tLongitud del vocabulario:\t{}\n\n'.format( format(len(vocabulary),',d') ) )

print ('\tCargando FastText Model...')
start_time_load = datetime.now()
ft = fasttext.load_model('/home/orlando/projects/data/embeddings/cc.es.300.bin')
fasttext.util.reduce_model(ft, 100)
end_time = datetime.now()
print('\tTiempo de carga: {}'.format(end_time - start_time_load))
print ('\tModelo cargado!\n')
print ( '\tEmbeddings de dimensión: \t{}'.format(ft.get_dimension()) )



fnsave = 'data/myembeddings100.txt'
data = 'word\tembeddings\n'
saveIntoFile(fnsave, data, 'w')

for i, word in enumerate(vocabulary):
	sys.stdout.write( '\tProcesando {} palabras...\r'.format( format(i,',d') ) )
	sys.stdout.flush()
	embed = []
	try:
		embed = ft.get_word_vector(word)
		embed = embed.tolist()
	except Exception as e:
		pass
	data = '{}\t{}\n'.format(word, embed)
	saveIntoFile(fnsave, data)

print ('\n\tTerminado!')

# -----------------------------------------------------------------------------------------------------------------
end_time = datetime.now()
print ('\n\t==========================')
print('\tDuration: {}'.format(end_time - start_time))
print ('\t==========================')
