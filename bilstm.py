class RNN(object):
	"""docstring for RelationExtraction"""

	def __init__(self, parameters, verbose=False):
		self.parameters = parameters
		self.verbose = verbose
		self.x_document = []
		self.y_document = []
		self.word_pad_token = 'padtoken'
		self.word_unk_token = 'unknowntoken'
		self.loadData()
		self.word2idx = self.getVocabulary() # self.getTagVocabulary(self.y_document, self.tag_pad_token, self.tag_unk_token)
		self.tag2idx = self.getTagVocabulary(self.y_document)
		self.words_len = len(self.word2idx)
		self.tags_len = len(self.tag2idx)
		print ( "\tSize vocabulary:\t{}".format( format(self.words_len,',d')) )
		print ( "\tSize categories:\t{}".format( format(self.tags_len,',d')) )
		self.split_train_test = 0.2
		self.random = 39
		self.X_train = []
		self.y_train = []
		self.X_test = []
		self.y_test = []
		self.prepareData()
		self.wordEmbeddings = []

	def loadData(self):
		"""
		input:
			id \t sentence \t relation # headers
		"""
		import sys
		import numpy as np
		categories = {}
		lengths = {}
		longitudes = []

		print ( '\tLoading documents, please wait.' )
		with open(self.parameters['corpus_path'], 'r') as f:
			for i, row in enumerate(f):
				if i == 0: continue # skip headers
				if self.verbose:
					sys.stdout.write( '\tLoading {} categories ...\r'.format( format(i, ',d') ) )
					sys.stdout.flush()
				category, id_doc, doc = row.replace('\n','').split('\t')
				if category not in categories: categories[category] = 1
		f.close()

		categories = dict(sorted(categories.items(), key=lambda x:x[0], reverse=False))
		categories = {cat:i for i, cat in enumerate(categories)}
		with open(self.parameters['corpus_path'], 'r') as f:
			for i, row in enumerate(f):
				if i == 0: continue # skip headers
				if self.verbose:
					sys.stdout.write( '\tLoading {} documents ...\r'.format( format(i, ',d') ) )
					sys.stdout.flush()
				# ---------------------------------------------------------------------
				category, id_doc, doc = row.replace('\n','').split('\t')
				self.x_document.append(doc.split())
				self.y_document.append(category)
				# ---------------------------------------------------------------------
				longitud =  len(doc.split())
				longitudes.append(longitud)
				if longitud not in lengths: lengths[longitud] = 1
				else: lengths[longitud] += 1

		f.close()

		lengths = dict( sorted(lengths.items(), key=lambda x:x[0], reverse=True) )
		# plot document histogram
		#  self.plotDocumentLengthHistogram(longitudes)

	def plotDocumentLengthHistogram(self, lengths):
		import matplotlib.pyplot as plt
		plt.figure(figsize=(18,8))
		plt.hist(lengths, bins="auto")
		plt.title('Documents histogram', fontsize=14)
		plt.ylabel('Documents', fontsize=12)
		plt.xlabel('Length', fontsize=12)
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.tight_layout()
		plt.grid()
		plt.savefig('img/docs_histogram.png', dpi=200)
		# plt.savefig('img/docs_histogram.eps', dpi=200)
		plt.show()

	def getVocabulary(self):
		import sys
		tmp_vocabulary = {}
		if self.verbose: print ('\n\tChecking vocabulary')
		for i, sentence in enumerate(self.x_document):
			if self.verbose:
				sys.stdout.write('\tprocesing {} documents ...\r'.format( format(i, ',d')))
				sys.stdout.flush()
			for word in sentence:
				if word not in tmp_vocabulary: tmp_vocabulary[word] = 1
				else: tmp_vocabulary[word] += 1
		if self.verbose: print ('\n\tSorting vocabulary')
		tmp_vocabulary = dict( sorted(tmp_vocabulary.items(), key=lambda x:x[1], reverse=True) )
		if parameters['vocabulary']['cut'] != 0:
			tmp_vocabulary = list(tmp_vocabulary.items())
			del tmp_vocabulary[ parameters['vocabulary']['cut']:]
			tmp_vocabulary = dict(tmp_vocabulary)

		vocab = {self.word_pad_token:0, self.word_unk_token:1}
		vocab.update( {item: i+2 for i, item in enumerate(tmp_vocabulary)} )
		del tmp_vocabulary
		return vocab

	def getWord2Idx(self):
		"""Extract words/tags dictionary"""
		vocab = {self.word_pad_token:0, self.word_unk_token:1}
		vocab.update( {item: i+2 for i, item in enumerate(self.y_document)} )
		return vocab

	def getTag2Idx(self):
		"""Extract words/tags dictionary"""
		vocab = {item: i for i, item in enumerate(self.y_document)}
		return vocab

	def getTagVocabulary(self, sentences):
		"""Extract tags dictionary"""
		vocab = {item: i for i, item in enumerate(set(tg for tg in sentences))}
		return vocab

	def saveVocabulary(self, filepath, vocabulary):
		"""
		Save vocabulary to make future predictions
		"""
		import numpy as np
		np.save( filepath, list(vocabulary.items()) )
		print ('\tVocabulary saved in: {}'.format(filepath))

	def loadEmbeddings(self):
		"""
		"""
		import numpy as np
		import sys
		from datetime import datetime
		start_time = datetime.now()

		print ( '\tLoading embeddings, please wait.' )
		# vector = np.zeros(300)  # zero vector for 'padding_token' word
		# self.wordEmbeddings.append(vector)

		tmpEmbeddings = {}
		with open(self.parameters['embed_path'], 'r') as f:
			for i, line in enumerate(f):
				if i==0: continue # skip headers
				if self.verbose:
					sys.stdout.write( '\tLoading {} embeddings ...\r'.format( format(i, ',d') ) )
					sys.stdout.flush()
				word, str_emb = line.replace('\n','').split('\t')
				if word not in tmpEmbeddings:
					# vector = np.array([float(num) for num in str_emb.split()])
					vector = np.array( eval(str_emb) )
					tmpEmbeddings[word] = vector
			f.close()

		for word in self.word2idx:
			if word in tmpEmbeddings:
				self.wordEmbeddings.append( tmpEmbeddings[word] )
				# countOK += 1
			else:
				print ("==> ", word, " <==")
				if word == self.word_pad_token:
					self.wordEmbeddings.append( tmpEmbeddings[self.word_pad_token] )
				elif word != self.word_pad_token:
					# countNO += 1
					self.wordEmbeddings.append( tmpEmbeddings[self.word_unk_token] )
					# vector = np.random.uniform(-0.25, 0.25, 300) # vector among -0.25 and 0.25 for "unknown_token" word inside
					# self.wordEmbeddings.append(vector)

		self.wordEmbeddings = np.array(self.wordEmbeddings)
		end_time = datetime.now()
		print('\tEmbeddings loaded in : {}\n'.format(end_time - start_time))

	def prepareData(self):
		"""
		method to process training and test data (words, tags, chars)
		"""
		print ('')
		from keras.preprocessing.sequence import pad_sequences
		from sklearn.model_selection import train_test_split
		from keras.utils import to_categorical
		import numpy as np

		from sklearn.preprocessing import LabelBinarizer, LabelEncoder

		X_snt = [[self.word2idx[w] if w in self.word2idx else self.word2idx[self.word_unk_token] for w in s] for s in self.x_document]
		y_tag = [[self.tag2idx[t]] for t in self.y_document]

		X_snt = pad_sequences(maxlen=self.parameters['max_doc_len'], sequences=X_snt, padding='post', value=self.word2idx[self.word_pad_token])
		y_tag = to_categorical(y_tag, self.tags_len)

		print ("\tRandom:\t", self.random)
		print ("\tTest size:\t", self.split_train_test)

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_snt, y_tag, test_size=self.split_train_test, random_state=self.random)

		self.X_train = np.array(self.X_train)
		self.X_test = np.array(self.X_test)
		self.y_train = np.array(self.y_train)
		self.y_test = np.array(self.y_test)

		print ('\n\tWords: {}\t{}'.format(self.X_train.shape, self.X_test.shape) )
		print ('\tTags: {}\t{}\n'.format(self.y_train.shape, self.y_test.shape))

	def compileModel(self):
		"""Compile Model"""
		from keras.optimizers import RMSprop
		from keras.utils import plot_model
		from keras.models import Sequential
		from keras import layers

		self.model = Sequential()
		self.model.add(layers.Embedding(input_dim=self.words_len, output_dim=self.parameters['words_embedding_size'], input_length=self.parameters['max_doc_len'], weights=[self.wordEmbeddings], trainable=False))
		self.model.add(layers.Bidirectional(layers.LSTM(self.parameters['words_units'], dropout=self.parameters['words_dropout'], recurrent_dropout=self.parameters['words_recurrent_dropout'], return_sequences=False), name="BiLSTM"))
		self.model.add(layers.Dense(self.tags_len, activation='softmax', name="Softmax"))

		opt = RMSprop(lr=self.parameters['learn_rate'])
		self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
		print ( self.model.summary() )
		plot_model(self.model, to_file='{}.png'.format(self.parameters['plot_model']['name']), dpi=self.parameters['plot_model']['dpi'], expand_nested=True, show_shapes=True )

	def fitModel(self):
		"""Fit model"""
		import numpy as np

		history = self.model.fit(
			self.X_train,
			self.y_train,
			batch_size = self.parameters['batch_size'],
			epochs = self.parameters['epochs'],
			validation_split = self.parameters['validation_split'],
			verbose = self.parameters['fit_verbose']
		)
		return history

	def saveModel(self, jsonpath, weightspath):
		"""
		Save model
		"""
		model_json = self.model.to_json()
		with open(jsonpath, "w") as json_file:
		    json_file.write(model_json)
		self.model.save_weights(weightspath)
		print ('\tModel saved in: {}'.format(jsonpath))
		print ('\tModel saved in: {}'.format(weightspath))

	def saveIntoFile(self, fname, data, mode='a'):
		"""Save your data into file"""
		g = open(fname, mode)
		g.write(data)
		g.close()

	def plotFunctions(self, history, name):
		"""Plot functions"""
		import pandas as pd
		import matplotlib.pyplot as plt

		hist = pd.DataFrame(history.history)

		plt.figure(figsize=(18,8))
		plt.subplot(1, 2, 1)
		tl = plt.plot(hist["accuracy"], color='blue', label='train') # marker='o',
		vl = plt.plot(hist["val_accuracy"], color='red', label='validation') #  marker='s',
		plt.title('Accuracy: train vs validation')
		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')
		plt.legend([tl[0], vl[0]], ['train', 'validation'], loc='best')
		plt.tight_layout()
		plt.grid()

		plt.subplot(1, 2, 2)
		tl = plt.plot(hist["loss"], color='blue', label='train') # marker='o',
		vl = plt.plot(hist["val_loss"], color='red', label='validation') #  marker='s',
		plt.title('Loss: train vs validation')
		plt.ylabel('Loss')
		plt.xlabel('Epochs')
		plt.legend([tl[0], vl[0]], ['train', 'validation'], loc='best')
		plt.tight_layout()
		plt.grid()
		plt.savefig('{}.png'.format(name), dpi=200)
		plt.savefig('{}.eps'.format(name), dpi=200)

	def getPredictions(self, predict_path, true_path):
		"""Get predictions"""
		from sklearn.metrics import classification_report as cr_individual
		from seqeval.metrics import classification_report as cr_compose
		import numpy as np

		print ('\tWorking in Predictions ...')
		predictions = self.model.predict(self.X_test, verbose=0)
		# ---------------------------------------------------------------------------------
		idx2tag = {i: w for w, i in self.tag2idx.items()}
		def pred2label(pred, the_path):
			out = []
			for i, pred_i in enumerate(pred):
				out_i = []
				out_i = np.argmax(pred_i)
				out.append( idx2tag[out_i] )
			return out

		y_pred = pred2label( predictions, predict_path )
		y_true = pred2label( self.y_test, true_path )

		labels = list( self.tag2idx.keys() ).copy()
		metric1 = cr_individual(y_true, y_pred, labels=labels)
		print (metric1)
		print ('-'*100)


from datetime import datetime
start_general_time = datetime.now()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# run code to get text classification
import warnings
import time

inicio = time.time()
warnings.filterwarnings('ignore')

embed_path = 'data/embeddings_corpora_100.txt'
corpus_path = 'data/mxnc_o.txt'
# corpus_path = 'data/mxnc_b.txt'
funcs_path = 'img/acc_loss'

# words_vocab = 'data/vocab/words_vocab.npy'
# tags_vocab = 'data/vocab/tags_vocab.npy'
# json_model = 'model/nn_model.json'
# weights_model = 'model/nn_model.h5'

parameters = {
	'corpus_path': corpus_path,
	'embed_path': embed_path,
	'max_doc_len': 500,
	'vocabulary': {
		'cut': 0,
	},
	'words_embedding_size': 100,
	'words_units': 100,
	'words_dropout': 0.4,
	'words_recurrent_dropout': 0.4,
	'learn_rate': 0.001,
	'summary': True,
	'plot_model': {
		'plot': True,
		'name': 'img/model',
		'dpi': 200,
	},
	'batch_size': 100,
	'epochs': 30,
	'validation_split': 0.2,
	'fit_verbose': 2
}
print ( '\tparameters used\n' )
for par in parameters:
	print ( '\t{}\t{}'.format(par, parameters[par]) )
print ('')
nn = RNN(parameters)
nn.loadEmbeddings()
nn.compileModel()
history = nn.fitModel()
nn.plotFunctions(history, funcs_path)
nn.getPredictions(pred_path, true_path)

print ('\n\n')
print ('='*50)
end_general_time = datetime.now()
print('\tTraining finished in: {}\n'.format(end_general_time - start_general_time))
print ('='*50)

