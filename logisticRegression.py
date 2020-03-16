from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
import numpy as np

from datetime import datetime
start_time = datetime.now()

categories = []
documents = []
classes = {}

filename = 'data/mxnc_o.txt'
# filename = 'data/mxnc_b.txt'

print ('\t{}'.format(filename))
with open(filename, 'r') as f:
	for i, row in enumerate(f):
		if i==0: continue # skip headers
		sys.stdout.write('\tprocesando {} documentos ...\r'.format( format(i, ',d') ))
		sys.stdout.flush()
		# category, doc = row.replace('\n','').split('\t') # using original documents without remove duplicate sentences
		category, id_doc, doc = row.replace('\n','').split('\t') # removing duplicate sentences. doc == sentence
		categories.append(category)
		documents.append(doc)
		if category not in classes: classes[category] = 1
		else: classes[category] += 1
f.close()
print ('\n\tData loaded\n')

X_train, X_test, y_train, y_test = train_test_split(documents, categories, test_size=0.2, random_state=39)

print ( '\t{}\t\tX_train'.format( format(len(X_train),',d') ) )
print ( '\t{}\t\ty_train'.format( format(len(y_train),',d') ) )
print ( '\t{}\t\tX_test'.format( format(len(X_test),',d') ) )
print ( '\t{}\t\ty_test'.format( format(len(y_test),',d') ) )
print ('')
# -----------------------------------------------------------------------------------------------
logreg = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(solver='lbfgs', max_iter=100, multi_class='auto', n_jobs=8, C=1e5)), ])
logreg.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = logreg.predict(X_test)

print('\naccuracy {}\n'.format( accuracy_score(y_pred, y_test) ) )
print( classification_report(y_test, y_pred, target_names=list(classes.keys())) )

# -----------------------------------------------------------------------------------------------------------------
end_time = datetime.now()
print ('\n\t==========================')
print('\tDuration: {}'.format(end_time - start_time))
print ('\t==========================')
