import sys
from datetime import datetime
start_time = datetime.now()

def saveIntoFile(fname, data, mode='a'):
	"""Save your data into file"""
	g = open(fname, mode)
	g.write(data)
	g.close()

def splitSpanishSentences(text):
	"""
	Requirements
	pip install -U nltk
	nltk.download('punkt')
	-------------------------------
	return sentences list of the text
	"""
	import nltk
	spanish_tokenizer = nltk.data.load('tokenizers/punkt/PY3/spanish.pickle')
	sentences = spanish_tokenizer.tokenize(text)
	return sentences

def cleanText(text):
    """Preproccess document and clean"""
    from nltk.corpus import stopwords
    import re

    text = text.lower() # lowercase text
    text = re.sub('[^a-zñáéíóöúü ]', ' ', text)
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if word not in set(stopwords.words('spanish')) ) # remove stopwords
    return text

categories = {
	'crimen': 'crimen.txt',
	'cultura': 'cultura.txt',
	'deportes': 'deportes.txt',
	'economia': 'economia.txt',
	'espectaculos': 'espectaculos.txt',
	'finanzas': 'finanzas.txt',
	'gossip': 'gossip.txt',
	'justicia': 'justicia.txt',
	'policia': 'policia.txt',
	'politica': 'politica.txt',
	'salud': 'salud.txt',
	'seguridad': 'seguridad.txt',
	'show': 'show.txt',
	'sociedad': 'sociedad.txt',
	'tecnologia': 'tecnologia.txt',
}


fnsave = 'data/sentences_corpora.txt'
data = 'category\tid_doc\tsentence\n'
saveIntoFile(fnsave, data, 'w')


for cat in categories:
	filename = '/home/orlando/projects/filtrado/data/categories/corpus/{}'.format( categories[cat] )
	# filename = 'data/corpus/{}'.format( categories[cat] )
	print ( "\tProcesando corpus de:\t{}".format(cat) )
	with open(filename, 'r') as f:
		for i, row in enumerate(f):
			if i==0: continue # skip headers
			# sys.stdout.write( '\tProcesando {} documentos...\r'.format( format(i), ',d') )
			# sys.stdout.flush()
			category, doc = row.replace('\n','').split('\t')
			sentences = splitSpanishSentences(doc)
			for snt in sentences:
				snt = cleanText(snt)
				data = '{}\t{}\t{}\n'.format(cat, i, snt)
				saveIntoFile(fnsave, data)
	f.close()
	print ('\n\tTerminado!\n\n')
print ('\n\tTerminado!')

# -----------------------------------------------------------------------------------------------------------------
end_time = datetime.now()
print ('\n\t==========================')
print('\tDuration: {}'.format(end_time - start_time))
print ('\t==========================')

