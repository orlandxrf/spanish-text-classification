def plot3Bars(data):
	import matplotlib.pyplot as plt
	import numpy as np


	index = np.arange(len(data))

	plt.figure(figsize=(15,8))
	ax = plt.subplot(111)

	labels = ['original', 'total', 'duplicates']

	w = 0.3
	for i, category in enumerate(data):
		ax.bar(i-0.3, data[category]['original'], width=w, color='tab:green', align='center', label=labels[0])
		ax.bar(i, data[category]['total'], width=w, color='tab:blue', align='center', label=labels[1])
		ax.bar(i+0.3, data[category]['duplicates'], width=w, color='tab:red', align='center', label=labels[2])
	plt.xticks(index, list(data.keys()), fontsize=12, rotation=30)
	plt.yticks(fontsize=12)
	plt.xlabel('Document categories', fontsize=14)
	plt.title('Total sentences number by category', fontsize=14)
	plt.legend(labels=labels, loc='best', fontsize=12)
	plt.tight_layout()
	plt.grid()
	plt.savefig('img/sentences.png')
	plt.show()




def plotLines(data):
	import matplotlib.pyplot as plt
	import numpy as np

	index = np.arange(len(data))

	labels = ['original', 'duplicates', 'total']

	ori = [ data[item]['original'] for item in data]
	dup = [ data[item]['duplicates'] for item in data]
	tot = [ data[item]['total'] for item in data]

	plt.figure(figsize=(15,8))
	plt.plot(index, ori, '-go')
	plt.plot(index, dup, '-ro')
	plt.plot(index, tot, '-bo')
	plt.xticks(index, list(data.keys()), fontsize=12, rotation=30)
	plt.yticks(fontsize=12)
	plt.xlabel('Document categories', fontsize=14)
	plt.title('Total sentences number by category', fontsize=14)
	plt.legend(labels=labels, loc='best', fontsize=12)
	plt.tight_layout()
	plt.grid()
	# plt.savefig('img/categories.png')
	plt.show()



naive = {
	'crime': {'pr': 0.00, 're': 0.00, 'f1': 0.00},
	'culture': {'pr': 0.84, 're': 0.69, 'f1': 0.76},
	'economy': {'pr': 0.71, 're': 0.67, 'f1': 0.69},
	'finance': {'pr': 1.00, 're': 0.05, 'f1': 0.09},
	'gossip': {'pr': 0.00, 're': 0.00, 'f1': 0.00},
	'health': {'pr': 0.88, 're': 0.58, 'f1': 0.70},
	'justice': {'pr': 0.96, 're': 0.12, 'f1': 0.22},
	'police': {'pr': 0.56, 're': 0.98, 'f1': 0.71},
	'politics': {'pr': 0.54, 're': 0.86, 'f1': 0.67},
	'security': {'pr': 0.76, 're': 0.08, 'f1': 0.14},
	'show': {'pr': 0.94, 're': 0.20, 'f1': 0.33},
	'society': {'pr': 0.66, 're': 0.28, 'f1': 0.39},
	'spectacle': {'pr': 0.66, 're': 0.90, 'f1': 0.76},
	'sports': {'pr': 0.80, 're': 0.96, 'f1': 0.88},
	'technology': {'pr': 0.93, 're': 0.35, 'f1': 0.51},
}

svm = {
	'crime': {'pr': 0.94, 're': 0.11, 'f1': 0.19},
	'culture': {'pr': 0.78, 're': 0.74, 'f1': 0.76},
	'economy': {'pr': 0.67, 're': 0.78, 'f1': 0.72},
	'finance': {'pr': 0.91, 're': 0.13, 'f1': 0.23},
	'gossip': {'pr': 0.60, 're': 0.01, 'f1': 0.02},
	'health': {'pr': 0.76, 're': 0.80, 'f1': 0.78},
	'justice': {'pr': 0.66, 're': 0.22, 'f1': 0.33},
	'police': {'pr': 0.64, 're': 0.96, 'f1': 0.77},
	'politics': {'pr': 0.64, 're': 0.80, 'f1': 0.71},
	'security': {'pr': 0.76, 're': 0.35, 'f1': 0.48},
	'show': {'pr': 0.96, 're': 0.71, 'f1': 0.82},
	'society': {'pr': 0.69, 're': 0.29, 'f1': 0.41},
	'spectacle': {'pr': 0.79, 're': 0.87, 'f1': 0.83},
	'sports': {'pr': 0.82, 're': 0.97, 'f1': 0.89},
	'technology': {'pr': 0.80, 're': 0.58, 'f1': 0.67},
}


lgr = {
	'crime': {'pr': 0.95, 're': 0.92, 'f1': 0.94},
	'culture': {'pr': 0.83, 're': 0.84, 'f1': 0.84},
	'economy': {'pr': 0.83, 're': 0.83, 'f1': 0.83},
	'finance': {'pr': 0.83, 're': 0.80, 'f1': 0.82},
	'gossip': {'pr': 0.92, 're': 0.82, 'f1': 0.87},
	'health': {'pr': 0.83, 're': 0.83, 'f1': 0.83},
	'justice': {'pr': 0.67, 're': 0.65, 'f1': 0.66},
	'police': {'pr': 0.92, 're': 0.94, 'f1': 0.93},
	'politics': {'pr': 0.75, 're': 0.76, 'f1': 0.75},
	'security': {'pr': 0.82, 're': 0.82, 'f1': 0.82},
	'show': {'pr': 0.97, 're': 0.97, 'f1': 0.97},
	'society': {'pr': 0.53, 're': 0.54, 'f1': 0.54},
	'spectacle': {'pr': 0.93, 're': 0.95, 'f1': 0.94},
	'sports': {'pr': 0.97, 're': 0.96, 'f1': 0.97},
	'technology': {'pr': 0.84, 're': 0.82, 'f1': 0.83},
}


