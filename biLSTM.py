class BiLSTM(object):
	"""docstring for RelationExtraction"""

	def __init__(self, max_snt_len, corpus_path, filepathPOS, verbose=False):
		self.verbose = verbose
		self.corpus_path = corpus_path
		self.pos_corpus_path = filepathPOS
		self.lower_case = False
		self.relation_codes = {}
		self.x_sentence = []
		self.y_sentence = []
		self.pos_sentence = [] # PoS (Part of Speech)
		self.word_pad_token = 'padtoken'
		self.word_unk_token = 'unknowntoken'
		self.tag_pad_token = 'O'
		self.tag_unk_token = 'UNK'
		self.char_pad_token = 'PAD'
		self.char_unk_token = 'UNK'
		self.max_snt_len = max_snt_len # 40 # 50 to process alias corpus
		self.max_char_len = 10
		self.max_case_len = 10
		self.max_pos_len = 10
		self.tag2idx = {} # self.getTagVocabulary(self.y_sentence, self.tag_pad_token, self.tag_unk_token)
		self.loadData()
		self.word2idx = self.getWordVocabulary(self.x_sentence, self.word_pad_token, self.word_unk_token)
		# self.readPOSCorpus()
		# self.pos2idx = self.getWordVocabulary(self.pos_sentence, self.tag_pad_token, self.tag_unk_token)
		# self.char2idx = self.getCharVocabulary(self.word2idx, self.char_pad_token, self.char_unk_token)
		# self.case2idx = {'numeric':2,'allLower':3,'allUpper':4,'initialUpper':5,'mainly_numeric':6,'contains_digit':7,self.word_pad_token:0,self.word_unk_token:1}
		self.words_len = len(self.word2idx)
		self.tags_len = len(self.tag2idx)
		# self.pos_len = len(self.pos2idx)
		# self.chars_len = len(self.char2idx)
		# self.case_len = len(self.case2idx)
		self.split_train_test = 0.2
		self.random = 39
		self.X_train = []
		# self.X_train_char = []
		# self.X_train_case = []
		self.y_train = []
		self.X_test = []
		# self.X_test_char = []
		# self.X_test_case = []
		self.y_test = []
		self.parameters = {}
		# self.model = None
		# self.caseEmbeddings = []
		self.prepareData()
		self.wordEmbeddings = []

	def loadData(self):
		"""
		input:
			id \t sentence \t relation # headers
		"""
		# import sys
		# import nlptools as nlp
		# tool = nlp.Tools()
		# data = 'id\tpos_sentence\trelation'
		# tool.saveData(self.pos_corpus_path, data, 'w')
		import sys
		import numpy as np
		categories = {}
		minimum = {'min':1000, 'id_doc':0}
		maximum = {'max':0, 'id_doc':0}
		lengths = {}
		stop = 50001
		longitudes = []

		print ( '\n\tLoading documents ...' )
		with open(self.corpus_path,'r') as f:
			for i, row in enumerate(f):
				if i == 0: continue # skip headers
				# if i==stop: break
				if self.verbose:
					sys.stdout.write( '\tLoading {} categories ...\r'.format( format(i, ',d') ) )
					sys.stdout.flush()
				category, id_doc, doc = row.replace('\n','').split('\t')
				if category not in categories: categories[category] = 1
		f.close()

		categories = dict(sorted(categories.items(), key=lambda x:x[0], reverse=False))
		categories = {cat:i for i, cat in enumerate(categories)}
		self.tag2idx = categories.copy()
		with open(self.corpus_path,'r') as f:
			for i, row in enumerate(f):
				if i == 0: continue # skip headers
				# if i==stop: break
				if self.verbose:
					sys.stdout.write( '\tLoading {} documents ...\r'.format( format(i, ',d') ) )
					sys.stdout.flush()
				# ---------------------------------------------------------------------
				category, id_doc, doc = row.replace('\n','').split('\t')
				tags = [0] * len(categories)
				tags[ categories[category] ] = 1
				self.x_sentence.append(doc.split()) # shape = (#total documents, different lengths) necessary set max_length (to padding and/or to cut)
				self.y_sentence.append(tags) # shape = (#total_documents, #total_classes)
				# ---------------------------------------------------------------------
				longitud =  len(doc.split())
				longitudes.append(longitud)
				if longitud not in lengths: lengths[longitud] = 1
				else: lengths[longitud] += 1

				if len(doc.split()) < minimum['min']:
					minimum['min'] = len(doc.split())
					minimum['id_doc'] = id_doc
				if len(doc.split()) > maximum['max']:
					maximum['max'] = len(doc.split())
					maximum['id_doc'] = id_doc

		f.close()
		# print ('')

		# ---------------------------------------------------------------------
		# self.x_sentence = np.array(self.x_sentence)
		# self.y_sentence = np.array(self.y_sentence)
		# print ('')
		# print ( '\t{}\t{}'.format(self.x_sentence.shape, self.y_sentence.shape) )

		# print (self.y_sentence)
		# ---------------------------------------------------------------------

		# print ('\n\tTerminado!\n')
		# for i, cat in enumerate(categories):
		# 	print ( '\t{}\t{}'.format(i+1, cat) )
		# print ('')
		# print ( '\tLongitud mínima de {} en el documento {}'.format(minimum['min'], minimum['id_doc']) )
		# print ( '\tLongitud máxima de {} en el documento {}\n\n'.format(maximum['max'], maximum['id_doc']) )

		# lengths = dict( sorted(lengths.items(), key=lambda x:x[0], reverse=True) )
		# for i, x in enumerate(lengths):
		# 	print ( '{}\t{}\t{}'.format(i+1, x, format(lengths[x]), ',d') )
		# self.plotDocumentLengthHistogram(longitudes)
		# exit(0)

	def plotDocumentLengthHistogram(self, lengths):
		import matplotlib.pyplot as plt
		plt.figure(figsize=(18,8))
		plt.hist(lengths, bins="auto")
		plt.title('Documents histogram', fontsize=14)
		plt.ylabel('Documents', fontsize=12)
		plt.xlabel('Length', fontsize=12)
		plt.tight_layout()
		plt.grid()
		plt.savefig('img/docs_histogram.png', dpi=200)
		plt.savefig('img/docs_histogram.eps', dpi=200)
		plt.show()

	def readPOSCorpus(self):
		with open(self.pos_corpus_path,'r') as f:
			for i, row in enumerate(f):
				if i == 0: continue # skip headers
				row = row.replace('\n', '').split('\t')
				sentence = [ (word.split('_')[0]) if '_' in word else word for word in row[1].split() ]
				self.pos_sentence.append( sentence )
		f.close()

	def __setSchema(self, ysentence, positions, entity, relation, tag):
		"""
		"""
		schema = { # IOBES schema. 'O' it's necessary
			1: ['S'],
			2: ['B', 'E'],
			3: ['B', 'I', 'E'],
		}

		if len(entity) == 1:
			ysentence[positions[0]] = '{}-{}-{}'.format(schema[1][0], tag.split('-')[1], relation ) # tag[-1:]
		if len(entity) == 2:
			tmplst = []
			for i, w in enumerate(entity):
				tmplst.append( '{}-{}-{}'.format(schema[2][i], tag.split('-')[1], relation ) ) # tag[-1:]
			ysentence[positions[0]:positions[1]] = tmplst
		if len(entity) > 2:
			tmplst = []
			for i, w in enumerate(entity):
				if i == 0:
					tmplst.append( '{}-{}-{}'.format(schema[3][0], tag.split('-')[1], relation ) ) # tag[-1:]
				elif i > 0 and i < (len(entity)-1):
					tmplst.append( '{}-{}-{}'.format(schema[3][1], tag.split('-')[1], relation ) ) # tag[-1:]
				else:
					tmplst.append( '{}-{}-{}'.format(schema[3][2], tag.split('-')[1], relation ) ) # tag[-1:]
			ysentence[positions[0]:positions[1]] = tmplst
		return ysentence

	def makeYvalues(self, sentence, relation, relation_codes):
		"""
		method to make y vectors from sentence X
		"""
		relation, positions = relation.split('~')
		positions = positions.split(',')
		positions.pop()
		en1, pos1 = positions[0].split(';')
		en2, pos2 = positions[1].split(';')
		pos1 = [int(p) for p in pos1.split(':')]
		pos2 = [int(p) for p in pos2.split(':')]

		entity1 = sentence[ pos1[0]:pos1[1] ]
		entity2 = sentence[ pos2[0]:pos2[1] ]

		ysentence = ['O'] * len(sentence)

		relation = '_'.join(relation.split('-'))
		ysentence = self.__setSchema(ysentence, pos1, entity1, relation, en1)
		ysentence = self.__setSchema(ysentence, pos2, entity2, relation, en2)
		return (sentence, ysentence)

	def getWordVocabulary(self, sentences, pad_token, unk_token):
		"""Extract words/tags dictionary"""
		vocab = {pad_token:0, unk_token:1}
		vocab.update( {item: i+2 for i, item in enumerate(set (w for s in sentences for w in s))} )
		return vocab

	def getTagVocabulary(self, sentences, pad_token, unk_token):
		"""Extract words/tags dictionary"""
		# vocab = {pad_token:0} # unk_token:1
		vocab = {item: i for i, item in enumerate(set (w for s in sentences for w in s))}
		return vocab

	def getCharVocabulary(self, words_dict, pad_token, unk_token):
		"""Extract char dictionary"""
		vocab = {pad_token:0, unk_token:1}
		vocab.update( {item: i+2 for i, item in enumerate(set (c for word in words_dict for c in word))} )
		return vocab

	def getCharFeatures(self, word):
		char_seq = [0] * self.max_char_len
		char_seq[0] = self.char2idx.get(word[0], self.char2idx[self.char_unk_token])
		if len(word) > 1: char_seq[5] = self.char2idx.get(word[-1], self.char2idx[self.char_unk_token])
		else: char_seq[5] = self.char2idx[self.char_pad_token]
		return char_seq

	def getCasing(self, word, caseLookup):
		casing = self.word_unk_token

		numDigits = 0
		for char in word:
			if char.isdigit():
				numDigits += 1

		digitFraction = numDigits / float(len(word))

		if word.isdigit(): #Is a digit
			casing = 'numeric'
		elif digitFraction > 0.5:
			casing = 'mainly_numeric'
		elif word.islower(): #All lower case
			casing = 'allLower'
		elif word.isupper(): #All upper case
			casing = 'allUpper'
		elif word[0].isupper(): #is a title, initial char upper, then all lower
			casing = 'initialUpper'
		elif numDigits > 0:
			casing = 'contains_digit'
		return caseLookup[casing]

	def getCaseFeatures(self, token):
		"""
		Case features
		"""
		import numpy as np
		case_features = {
			# 'isalphanumeric':1 if token.isalnum() else 0, # if string has at least 1 character and all characters are alphanumeric
			# 'isalphabetic':1 if token.isalpha() else 0, # if string has at least 1 character and all characters are alphabetic
			'isdigit':1 if token.isdigit() else 0, # if string contains only digits
			'islower':1 if token.islower() else 0, # if string contains lowercase
			'isupper':1 if token.isupper() else 0, # if string contains uppercase
			'istitle':1 if token.istitle() else 0, # if string is title case
			self.word_pad_token:0,
			self.word_unk_token:0,
		}
		values = np.array( list( case_features.values() ) )
		print (values)
		one_hot = np.identity( len(case_features.keys()), dtype="float32" )[values]

		return one_hot

	def saveVocabulary(self, filepath, vocabulary):
		"""
		Save vocabulary to make future predictions
		"""
		import numpy as np
		np.save( filepath, list(vocabulary.items()) )
		print ('\tVocabulary saved in: {}'.format(filepath))

	def loadEmbeddings(self, embed_path):
		"""
		"""
		import numpy as np
		import sys
		from datetime import datetime
		start_time = datetime.now()

		print ( '\n\tLoading embeddings, please wait ...' )
		# vector = np.zeros(300)  # zero vector for 'padding_token' word
		# self.wordEmbeddings.append(vector)

		tmpEmbeddings = {}
		with open(embed_path, 'r') as f:
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
		from keras.preprocessing.sequence import pad_sequences
		from sklearn.model_selection import train_test_split
		from keras.utils import to_categorical
		import numpy as np

		X_snt = [[self.word2idx[w] for w in s] for s in self.x_sentence]
		# y_tag = [[self.tag2idx[w] for w in s] for s in self.y_sentence]

		# split or padding sentences using keras pad_sequences
		X_snt = pad_sequences(maxlen=self.max_snt_len, sequences=X_snt, padding='post', value=self.word2idx[self.word_pad_token])
		# y_tag = pad_sequences(maxlen=self.max_snt_len, sequences=y_tag, padding='post', value=self.tag2idx[self.tag_pad_token])
		# y_tag = [to_categorical(i, num_classes=self.tags_len) for i in self.y_sentence]

		self.y_sentence = np.array(self.y_sentence)
		# y_tag = np.array(y_tag)

		# print ( y_tag.shape )
		# print ( y_tag )
		# # exit(0)

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_snt, self.y_sentence, test_size=self.split_train_test, random_state=self.random)
		# self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_snt, y_tag, test_size=self.split_train_test, random_state=self.random)

		# self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_snt, y_tag, test_size=self.split_train_test, random_state=self.random)
		# self.X_train_char, self.X_test_char, _, _ = train_test_split(X_char, y_tag, test_size=self.split_train_test, random_state=self.random)
		# self.X_train_case, self.X_test_case, _, _ = train_test_split(X_case, y_tag, test_size=self.split_train_test, random_state=self.random)
		# self.X_train_pos, self.X_test_pos, _, _ = train_test_split(X_pos, y_tag, test_size=self.split_train_test, random_state=self.random)

		# syns experiment
		# Words: (40000, 250)	(10000, 250)
		# Tags: (40000, 3)	(10000, 3)

		# alias experiment
		print ('\tWords: {}\t{}'.format(self.X_train.shape, self.X_test.shape) )
		print ('\tTags: {}\t{}\n'.format(self.y_train.shape, self.y_test.shape))

	def compileModel(self):
		"""Compile Model"""
		from keras.initializers import RandomUniform
		from keras.optimizers import RMSprop
		from keras.utils import plot_model
		from keras.models import Model
		from keras import layers

		# print (self.wordEmbeddings.shape)
		# print (self.words_len)
		# print (self.caseEmbeddings.shape)
		# print (self.chars_len, self.max_char_len)
		# exit(0)

		# Words
		words_input = layers.Input(shape=(self.max_snt_len,), name="Words_input")
		# word_out = layers.Embedding(input_dim=self.words_len, output_dim=self.parameters['words_embedding_size'], input_length=self.max_snt_len, weights=[self.wordEmbeddings], trainable=False, name='Words_embeddings_300')(words_input)
		word_out = layers.Embedding(input_dim=self.words_len, output_dim=self.parameters['words_embedding_size'], input_length=self.max_snt_len, trainable=True, name='Words_embeddings_100')(words_input)

		# Casing features
		# case_input = layers.Input(shape=(self.max_snt_len, self.max_case_len), name='Case_input')
		# emb_case = layers.TimeDistributed(layers.Embedding(input_dim=self.case_len, output_dim=self.parameters['case_embedding_size'], input_length=self.max_case_len, mask_zero=True, trainable=True, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='Case_embeddings')(case_input)
		# case_out = layers.TimeDistributed(layers.LSTM(units=self.parameters['case_units'], return_sequences=False, dropout=self.parameters['case_dropout'], recurrent_dropout=self.parameters['case_recurrent_dropout']), name='LSTM_Case_layer')(emb_case)

		# # Part of Speech
		# pos_input = layers.Input(shape=(self.max_snt_len, self.max_pos_len), name='Part_of_Speech_input')
		# emb_pos = layers.TimeDistributed(layers.Embedding(input_dim=self.pos_len, output_dim=self.parameters['pos_embedding_size'], input_length=self.max_pos_len, mask_zero=True, trainable=True, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='PoS_embeddings')(pos_input)
		# pos_out = layers.TimeDistributed(layers.LSTM(units=self.parameters['pos_units'], return_sequences=False, dropout=self.parameters['pos_dropout'], recurrent_dropout=self.parameters['pos_recurrent_dropout']), name='LSTM_PoS_layer')(emb_pos)

		# # input and embeddings for characters
		# char_input = layers.Input(shape=(self.max_snt_len, self.max_char_len,), name='Character_input')
		# emb_char = layers.TimeDistributed(layers.Embedding(input_dim=self.chars_len, output_dim=self.parameters['chars_embedding_size'], input_length=self.max_char_len, trainable=True, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5) ), name='Character_embeddings_10')(char_input)

		# # character LSTM to get word encodings by Features
		# char_out = layers.TimeDistributed(layers.LSTM(units=self.parameters['chars_units'], return_sequences=False, dropout=self.parameters['chars_dropout'], recurrent_dropout=self.parameters['chars_recurrent_dropout']), name='LSTM_Character_layer')(emb_char)

		# # main LSTM
		# conca = layers.concatenate([word_out, case_out, pos_out, char_out])
		# conca = layers.SpatialDropout1D(0.3)(conca)

		# Bi LSTM layer
		# lstm_out = layers.Bidirectional(layers.LSTM(self.parameters['words_units'], return_sequences=True, dropout=self.parameters['words_dropout'], recurrent_dropout=self.parameters['words_recurrent_dropout']), name='Bi_LSTM_Words_layer')(conca)

		lstm_out = layers.Bidirectional(layers.LSTM(self.parameters['words_units'], return_sequences=True, dropout=self.parameters['words_dropout'], recurrent_dropout=self.parameters['words_recurrent_dropout']), name='Bi_LSTM_Words_layer')(word_out)

		# Dense layer
		output = layers.TimeDistributed(layers.Dense(self.tags_len, activation="softmax"), name='Softmax_layer')(lstm_out)

		# self.model = Model([words_input, case_input, pos_input, char_input], output)
		self.model = Model(words_input, output)
		opt = RMSprop(lr=self.parameters['learn_rate'], decay=0.0)
		self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])

		if self.parameters['summary']:
			print (self.model.summary())
		if self.parameters['plot_model']['plot']:
			plot_model(self.model, to_file='{}.png'.format(self.parameters['plot_model']['name']), dpi=self.parameters['plot_model']['dpi'], expand_nested=True, show_shapes=True )
			plot_model(self.model, to_file='{}.pdf'.format(self.parameters['plot_model']['name']), dpi=self.parameters['plot_model']['dpi'], expand_nested=True, show_shapes=True )

	def compileModel2(self):
		"""Compile Model"""
		from keras.initializers import RandomUniform
		from keras.optimizers import RMSprop
		from keras.utils import plot_model
		# from keras.models import Model
		from keras.models import Sequential
		from keras import layers

		self.model = Sequential()
		self.model.add(layers.Embedding(input_dim=self.words_len, output_dim=self.parameters['words_embedding_size'], input_length=self.max_snt_len, weights=[self.wordEmbeddings], trainable=False))
		self.model.add(layers.SpatialDropout1D(0.2))
		self.model.add(layers.LSTM(self.parameters['words_units'], dropout=self.parameters['words_dropout'], recurrent_dropout=self.parameters['words_recurrent_dropout'])) # return_sequences=True,
		self.model.add(layers.Dense(self.tags_len, activation='softmax'))
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		print ( self.model.summary() )
		plot_model(self.model, to_file='{}.png'.format(self.parameters['plot_model']['name']), dpi=self.parameters['plot_model']['dpi'], expand_nested=True, show_shapes=True )

	def fitModel(self):
		"""Fit model"""
		# print (self.parameters)
		import numpy as np
		# history = self.model.fit(
		# 	[
		# 		self.X_train,
		# 		np.array(self.X_train_case).reshape( (len(self.X_train_case), self.max_snt_len, self.max_case_len) ),
		# 		np.array(self.X_train_pos).reshape( (len(self.X_train_pos), self.max_snt_len, self.max_pos_len) ),
		# 		np.array(self.X_train_char).reshape( (len(self.X_train_char), self.max_snt_len, self.max_char_len) )
		# 	],
		# 	np.array(self.y_train).reshape(len(self.y_train), self.max_snt_len, 1),
		# 	batch_size = self.parameters['batch_size'],
		# 	epochs = self.parameters['epochs'],
		# 	validation_split = self.parameters['validation_split'],
		# 	verbose = self.parameters['fit_verbose']
		# )

		history = self.model.fit(
			self.X_train,
			self.y_train,
			batch_size = self.parameters['batch_size'],
			epochs = self.parameters['epochs'],
			validation_split = self.parameters['validation_split'],
			verbose = self.parameters['fit_verbose']
		)

		# history = self.model.fit(self.X_train,
		# 	np.array(self.y_train).reshape(len(self.y_train), self.max_snt_len, 1),
		# 	batch_size = self.parameters['batch_size'],
		# 	epochs = self.parameters['epochs'],
		# 	validation_split = self.parameters['validation_split'],
		# 	verbose = self.parameters['fit_verbose']
		# )

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
		# old tensorflow version
		tl = plt.plot(hist["acc"], color='blue', label='train') # marker='o',
		vl = plt.plot(hist["val_acc"], color='red', label='validation') #  marker='s',
		# new tensorflow version
		# tl = plt.plot(hist["accuracy"], color='blue', label='train') # marker='o',
		# vl = plt.plot(hist["val_accuracy"], color='red', label='validation') #  marker='s',
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

	def getPredictions(self, predict_path):
		"""Get predictions"""
		from sklearn.metrics import classification_report as cr_individual
		from seqeval.metrics import classification_report as cr_compose
		import numpy as np

		predictions = self.model.predict(self.X_test, verbose=0)

		# predictions = self.model.predict([self.X_test,
		# 	np.array(self.X_test_case).reshape((len(self.X_test_case), self.max_snt_len, self.max_case_len)),
		# 	np.array(self.X_test_pos).reshape((len(self.X_test_pos), self.max_snt_len, self.max_pos_len)),
		# 	np.array(self.X_test_char).reshape((len(self.X_test_char), self.max_snt_len, self.max_char_len))
		# ], verbose=0)

		# ---------------------------------------------------------------------------------
		# predicted data
		# test_pred = model.predict(X_te, verbose=0)
		idx2tag = {i: w for w, i in self.tag2idx.items()}
		def pred2label(pred):
			out = []
			for pred_i in pred:
				out_i = []
				for p in pred_i:
					p_i = np.argmax(p)
					out_i.append(idx2tag[p_i].replace('<pad>', 'O'))
				out.append(out_i)
			return out

		y_pred = pred2label( predictions )
		y_true = pred2label( self.y_test )

		labels = list( self.tag2idx.keys() ).copy()
		# labels.remove('O') # remove class 'O'
		# labels = sorted(labels, key=lambda x: x.split('-')[2]) # sort labels

		metric2 = cr_compose(y_true, y_pred)
		print ('-'*100)
		print (metric2)
		print ('-'*100)

		true_y, pred_y = [], []
		for k in range(len(y_true)): true_y += y_true[k]
		for k in range(len(y_pred)): pred_y += y_pred[k]
		metric1 = cr_individual(true_y, pred_y, labels=labels)
		print (metric1)
		print ('-'*100)
		# ---------------------------------------------------------------------------------


		# y_words, y_true, y_pred = [], [], []

		# idx2word = {self.word2idx[word]: word for word in self.word2idx}
		# idx2tag = {self.tag2idx[tag]: tag for tag in self.tag2idx}

		# g = open(predict_path, 'w') # reset file
		# g.close()
		# for i, __ in enumerate(predictions):
		# 	p = np.argmax(predictions[i], axis=-1)
		# 	tmpWord, tmpTrue, tmpPred = [], [], []
		# 	data = ''
		# 	for idxWord, idxTrue, idxPred in zip(self.X_test[i], self.y_test[i], p):
		# 		if idxWord != 0: # padding token
		# 			tmpWord.append(idx2word[idxWord])
		# 			tmpTrue.append(idx2tag[idxTrue])
		# 			tmpPred.append(idx2tag[idxPred])
		# 	data = '[{}\t{}\t{}]'.format(tmpWord, tmpTrue, tmpPred)
		# 	# tool.saveData(predict_path, data, 'a')
		# 	self.saveIntoFile(predict_path, data)

		# 	# y_words.append(tmpWord)
		# 	y_true.append(tmpTrue)
		# 	y_pred.append(tmpPred)

		# labels = list( self.tag2idx.keys() ).copy()
		# # labels.remove('O') # remove class 'O'
		# # labels = sorted(labels, key=lambda x: x.split('-')[2]) # sort labels

		# metric2 = cr_compose(y_true, y_pred)
		# print ('-'*100)
		# print (metric2)
		# print ('-'*100)

		# true_y, pred_y = [], []
		# for k in range(len(y_true)): true_y += y_true[k]
		# for k in range(len(y_pred)): pred_y += y_pred[k]
		# metric1 = cr_individual(true_y, pred_y, labels=labels)
		# print (metric1)
		# print ('-'*100)

relation_codes = {
	'Acronym': 'AC',
	'Appointed-By': 'AB',
	'Beat-To': 'BT',
	'Candidate-Job': 'CJ',
	'Candidate-Organization': 'CO',
	'Candidate-PoliticalParty': 'CP',
	'Candidate-To': 'CT',
	'Choose-By': 'CB',
	'Define-Posture': 'DP',
	'Dismissal': 'DI',
	'Human-Relationship': 'HR',
	'Investigate-By': 'IB',
	'Is-In': 'IN',
	'Job-Organization': 'JO',
	'Job-Person': 'JP',
	'Meeting': 'ME',
	'Member-Collection': 'MC',
	'Message': 'MS',
	'Old-Job': 'OJ',
	'Organization-Person': 'OP',
	'Proposal-By': 'PR',
	'Proposed-By': 'PB',
	'Resign': 'RS',
	'Support-BY': 'SB',
	'Work-Together': 'WT',
}

exp = 'alias' # 'alias' 'embed' 'syns'
increment = 1
max_snt_len = 50 # 40 or 50 # max sentence length. 50 to alias corpus. 40 for embed and syns.
get_base_path = '/home/orlando/projects/exp_{}'.format(exp)
set_base_path = '/home/orlando/projects/increment_features/{}/{}'.format(exp, increment)

# ---------------------------------------------------------------
# get corpus
words_corpus = '{}/data/{}_words_corpus.txt'.format(get_base_path, exp)
pos_corpus = '{}/data/{}_pos_corpus.txt'.format(get_base_path, exp)
embed_corpus = '{}/data/{}_embed_corpus.txt'.format(get_base_path, exp)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# plot model, loss and accuracy funtions
model_path = '{}/{}_model_25_{}'.format(set_base_path, exp, increment) # plot architecture model. To png and eps
funcs_path = '{}/{}_funcs_25_{}'.format(set_base_path, exp, increment) # acc and loss functions. To png and eps
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# save data for future predictions in unknow (new) data
words_vocab = '{}/model/{}_words_vocab_25_{}.npy'.format(set_base_path, exp, increment)
tags_vocab = '{}/model/{}_tags_vocab_25_{}.npy'.format(set_base_path, exp, increment)
pos_vocab = '{}/model/{}_pos_vocab_25_{}.npy'.format(set_base_path, exp, increment)
json_model = '{}/model/{}_model_25_{}.json'.format(set_base_path, exp, increment)
weights_model = '{}/model/{}_model_25_{}.h5'.format(set_base_path, exp, increment)
pred_path = '{}/{}_predictions_25_{}.txt'.format(set_base_path, exp, increment) # save true and pred data, they was used in metrics
# ---------------------------------------------------------------

# print (get_base_path)
# print (set_base_path)
# print (words_corpus)
# print (pos_corpus)
# print (embed_corpus)
# print (model_path)
# print (funcs_path)
# print (words_vocab)
# print (tags_vocab)
# print (pos_vocab)
# print (json_model)
# print (weights_model)
# print (pred_path)
# exit(0)

from datetime import datetime
start_general_time = datetime.now()

parameters = {
	'chars_embedding_size': 10,
	'chars_units': 64,
	'chars_dropout': 0.2,
	'chars_recurrent_dropout': 0.2,
	'case_embedding_size': 10,
	'case_units': 64,
	'case_dropout': 0.2,
	'case_recurrent_dropout': 0.2,
	'pos_embedding_size': 10,
	'pos_units': 64,
	'pos_dropout': 0.2,
	'pos_recurrent_dropout': 0.2,
	'words_embedding_size': 100,
	'words_units': 64,
	'words_dropout': 0.4,
	'words_recurrent_dropout': 0.4,
	'learn_rate': 0.01,
	'summary': True,
	'plot_model': {
		'plot': True,
		'name': 'img/modelo',
		'dpi': 200,
	},
	'batch_size': 128,
	'epochs': 20,
	'validation_split': 0.2,
	'fit_verbose': 2
}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# run code to get relation extraction
import warnings
import time

inicio = time.time()
warnings.filterwarnings('ignore')

max_snt_len = 250;
# embed_path = 'data/embeddings_corpora.txt'
embed_path = 'data/embeddings_corpora_100.txt'
corpus_path = 'data/news_corpora.txt'
filepathPOS = '/home/orlando/treetagger/'
pred_path = 'data/predictions.txt'
funcs_path = 'img/acc_loss'

words_vocab = 'data/vocab/words_vocab.npy'
tags_vocab = 'data/vocab/tags_vocab.npy'
json_model = 'model/nn_model.json'
weights_model = 'model/nn_model.h5'

nn = BiLSTM(max_snt_len, corpus_path, filepathPOS, verbose=False)
nn.parameters = parameters # set parameters to compile and fit model
nn.loadEmbeddings(embed_path)
print("\tSave vocabularies to disk")
nn.saveVocabulary(words_vocab, nn.word2idx)
nn.saveVocabulary(tags_vocab, nn.tag2idx)
nn.compileModel2()
history = nn.fitModel()
print("\tSave model to disk")
nn.saveModel(json_model, weights_model)
nn.plotFunctions(history, funcs_path)
nn.getPredictions(pred_path)

print ('\n\n')
print ('='*50)
end_general_time = datetime.now()
print('\tTraining finished in: {}\n'.format(end_general_time - start_general_time))



# re = RelationExtraction(max_snt_len, words_corpus, pos_corpus)
# # re.max_snt_len = max_snt_len # set max sentence length
# re.parameters = parameters # set parameters to compile and fit model
# re.relation_codes = relation_codes
# # print("\tSave vocabularies to disk")
# # re.saveVocabulary(words_vocab, re.word2idx)
# # re.saveVocabulary(tags_vocab, re.tag2idx)
# # re.saveVocabulary(pos_vocab, re.pos2idx)
# re.loadEmbeddings(embed_corpus, re.word2idx, re.word_pad_token)
# re.compileModel()
# history = re.fitModel()
# # print("\tSave model to disk")
# # re.saveModel(json_model, weights_model)
# re.plotFunctions(history, funcs_path)
# re.getPredictions(pred_path)

# fin = time.time()
# hours, rem = divmod(fin - inicio, 3600)
# minutes, seconds = divmod(rem, 60)
# print ( '\n\tTotal elapsed time {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds) )
# print ('\nFinished !')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ------------------------------------------------------------------------------
# Instrucciones para generar modelos
# Paso 0
# 	Oraciones con multiples relaciones etiquetadas, se duplicaron tantas veces como relaciones contiene
# 	entrada = relations_corpus.txt
# 	salida = corpus_ok.txt
# Paso 1
# 	Augmentar los datos con diferentes tecnicas:
# 	entrada = corpus_ok.txt
# 	A: Synonyms
# 		Generar nuevo corpus, convertir corpus normal a:
# 		a) convertir a minúsculas
# 		b) generar vocabulario de lemmas, (sin tomar en cuenta entidades nombradas y palabras vacías)
# 		c) reemplazar en las oraciones
# 		salida = corpus_syns_no_padding.txt
# 	B: Embeddings con FastText
# 		Generar vocabulario de palabras:
# 		a) cargar modelo de embeddings de FastText
# 		b) buscar las 5 palabras más similares
# 		c) reemplazar en las oraciones
# 		salida = corpus_with_embeddings2.txt
# 	C: Alias
# 		Generar vocabulario de entidades nombradas
# 		a) buscar en wikidata id de la entidad en wikidata
# 		b) en un nueva consulta con id obtener alias de la entidad
# 		c) reemplazar en las oraciones
# 		salida = corpus_with_alias.txt
# Paso 2
# 	Generar corpus de embeddings:
# 	entrada = corpus correspondiente (syns, embed, alias)
# 	A. Obtener vocabulario del nuevo corpus generado
# 	B. Generar corpus de embeddings
