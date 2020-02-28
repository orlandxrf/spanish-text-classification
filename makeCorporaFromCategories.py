import sys
from datetime import datetime
start_time = datetime.now()

def saveToFile(fname, data, mode='a'):
	g = open(fname, mode)
	g.write(data)
	g.close()

def clean_text(text):
    """Preproccess document and clean"""
    from nltk.corpus import stopwords
    import re

    text = text.lower() # lowercase text
    text = re.sub('[^a-zñáéíóöúü ]', ' ', text)
    text = ' '.join(text.split())
    text = ' '.join(word for word in text.split() if word not in set(stopwords.words('spanish')) ) # remove stopwords
    return text

def joinCorpus(catgeory, path_input, pat_output):
	with open(path_input, 'r') as f:
		for i, row in enumerate(f):
			if i == 0: continue # skip headers
			sys.stdout.write( '\tProcesando {} documentos...\r'.format( format(i, ',d') ) )
			sys.stdout.flush()
			id_doc, doc = row.replace('\n','').split('\t')
			data = '{}\t{}\t{}\n'.format(catgeory, id_doc, doc)
			saveToFile(fnsave, data)
	f.close()

categories = {
	'crime': 'crime.txt',
	'culture': 'culture.txt',
	'economy': 'economy.txt',
	'finance': 'finance.txt',
	'gossip': 'gossip.txt',
	'health': 'health.txt',
	'justice': 'justice.txt',
	'police': 'police.txt',
	'politics': 'politics.txt',
	'security': 'security.txt',
	'show': 'show.txt',
	'society': 'society.txt',
	'spectacle': 'spectacle.txt',
	'sports': 'sports.txt',
	'technology': 'technology.txt',
}

absolutepath = '/home/orlando/projects/filtrado/data/categories/data/clean'
fnsave = 'data/news_corpora.txt'
data = 'category\tid_doc\tdoc\n'
saveToFile(fnsave, data, 'w')

categories = dict( sorted(categories.items(), key=lambda x:x[0]) )

for i, cat in enumerate(categories):
	fname = '{}/{}'.format(absolutepath, categories[cat])
	print ( '\tProcesando\t{}\t{}\n'.format(i+1, cat) )
	joinCorpus(cat, fname, fnsave)

print ('\tTerminado')

# -----------------------------------------------------------------------------------------------------------------
end_time = datetime.now()
print ('\n\t==========================')
print('\tDuration: {}'.format(end_time - start_time))
print ('\t==========================')
