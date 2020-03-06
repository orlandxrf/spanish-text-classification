def plot3Bars(data, labels, metric):
	import matplotlib.pyplot as plt
	import numpy as np


	index = np.arange(len( [data[0][cat][metric] for cat in data[0]] ))

	plt.figure(figsize=(15,8))
	ax = plt.subplot(111)

	# labels = ['LR', 'SVM', 'NB']
	labels.reverse()

	w = 0.2
	for i, category in enumerate(data[0]):
		ax.bar(i-(w/2+w), data[3][category][metric], width=w, color='tab:green', align='center', label=labels[3])
		ax.bar(i-(w/2), data[2][category][metric], width=w, color='tab:orange', align='center', label=labels[2])
		ax.bar(i+(w/2), data[1][category][metric], width=w, color='tab:red', align='center', label=labels[1])
		ax.bar(i+(w/2+w), data[0][category][metric], width=w, color='tab:blue', align='center', label=labels[0])
	plt.xticks(index, list(data[0].keys()), fontsize=12, rotation=30)
	plt.yticks(fontsize=12)
	plt.xlabel('Categorías', fontsize=14)
	plt.ylabel('{} score'.format(metric), fontsize=14)
	# plt.title('Resultados de la métrica F1 sobre los clasificadores', fontsize=14)
	plt.legend(labels=labels, loc='best', fontsize=12)
	plt.tight_layout()
	# plt.grid()
	plt.savefig( 'img/{}_score_bars.png'.format(metric) )
	plt.savefig( 'img/{}_score_bars.eps'.format(metric) )
	plt.show()



def plotLines(data, labels, metric):
	import matplotlib.pyplot as plt
	import numpy as np

	index = np.arange(len( [data[0][cat][metric] for cat in data[0]] ))

	# labels = ['NB', 'SVM', 'LR']

	# ori = [ data[item]['precision'] for item in data]
	# dup = [ data[item]['recall'] for item in data]
	# tot = [ data[item]['f1-score'] for item in data]

	plt.figure(figsize=(15,8))
	plt.plot(index, [data[0][cat][metric] for cat in data[0]], 'tab:blue', marker='o')
	plt.plot(index, [data[1][cat][metric] for cat in data[1]], 'tab:red', marker='o')
	plt.plot(index, [data[2][cat][metric] for cat in data[2]], 'tab:orange', marker='o')
	plt.plot(index, [data[3][cat][metric] for cat in data[3]], 'tab:green', marker='o')
	# plt.plot(index, dup, '-ro')
	# plt.plot(index, tot, '-bo')
	plt.xticks(index, list(data[0].keys()), fontsize=12, rotation=30)
	plt.yticks(fontsize=12)
	plt.xlabel('Categorías', fontsize=14)
	plt.ylabel('{} score'.format(metric), fontsize=14)
	# plt.title('Total sentences number by category', fontsize=14)
	plt.legend(labels=labels, loc='best', fontsize=12)
	plt.tight_layout()
	# plt.grid()
	plt.savefig( 'img/{}_score_lines.png'.format(metric) )
	plt.savefig( 'img/{}_score_lines.eps'.format(metric) )
	plt.show()



naive = {
	'deportes': {'Precision': 0.80, 'Recall': 0.96, 'F1': 0.88},
	'policiaca': {'Precision': 0.56, 'Recall': 0.98, 'F1': 0.71},
	'espectáculos': {'Precision': 0.66, 'Recall': 0.90, 'F1': 0.76},
	'política': {'Precision': 0.54, 'Recall': 0.86, 'F1': 0.67},
	'seguridad': {'Precision': 0.76, 'Recall': 0.08, 'F1': 0.14},
	'sociedad': {'Precision': 0.66, 'Recall': 0.28, 'F1': 0.39},
	'cultura': {'Precision': 0.84, 'Recall': 0.69, 'F1': 0.76},
	'economía': {'Precision': 0.71, 'Recall': 0.67, 'F1': 0.69},
	'justicia': {'Precision': 0.96, 'Recall': 0.12, 'F1': 0.22},
	'show': {'Precision': 0.94, 'Recall': 0.20, 'F1': 0.33},
	'salud': {'Precision': 0.88, 'Recall': 0.58, 'F1': 0.70},
	'tecnología': {'Precision': 0.93, 'Recall': 0.35, 'F1': 0.51},
	'finanzas': {'Precision': 1.00, 'Recall': 0.05, 'F1': 0.09},
	'gossip': {'Precision': 0.00, 'Recall': 0.00, 'F1': 0.00},
	'crimen': {'Precision': 0.00, 'Recall': 0.00, 'F1': 0.00},
}

svm = {
	'deportes': {'Precision': 0.82, 'Recall': 0.97, 'F1': 0.89},
	'policiaca': {'Precision': 0.64, 'Recall': 0.96, 'F1': 0.77},
	'espectáculos': {'Precision': 0.79, 'Recall': 0.87, 'F1': 0.83},
	'política': {'Precision': 0.64, 'Recall': 0.80, 'F1': 0.71},
	'seguridad': {'Precision': 0.76, 'Recall': 0.35, 'F1': 0.48},
	'sociedad': {'Precision': 0.69, 'Recall': 0.29, 'F1': 0.41},
	'cultura': {'Precision': 0.78, 'Recall': 0.74, 'F1': 0.76},
	'economía': {'Precision': 0.67, 'Recall': 0.78, 'F1': 0.72},
	'justicia': {'Precision': 0.66, 'Recall': 0.22, 'F1': 0.33},
	'show': {'Precision': 0.96, 'Recall': 0.71, 'F1': 0.82},
	'salud': {'Precision': 0.76, 'Recall': 0.80, 'F1': 0.78},
	'tecnología': {'Precision': 0.80, 'Recall': 0.58, 'F1': 0.67},
	'finanzas': {'Precision': 0.91, 'Recall': 0.13, 'F1': 0.23},
	'gossip': {'Precision': 0.60, 'Recall': 0.01, 'F1': 0.02},
	'crimen': {'Precision': 0.94, 'Recall': 0.11, 'F1': 0.19},
}


lgr = {
	'deportes': {'Precision': 0.97, 'Recall': 0.96, 'F1': 0.97},
	'policiaca': {'Precision': 0.92, 'Recall': 0.94, 'F1': 0.93},
	'espectáculos': {'Precision': 0.93, 'Recall': 0.95, 'F1': 0.94},
	'política': {'Precision': 0.75, 'Recall': 0.76, 'F1': 0.75},
	'seguridad': {'Precision': 0.82, 'Recall': 0.82, 'F1': 0.82},
	'sociedad': {'Precision': 0.53, 'Recall': 0.54, 'F1': 0.54},
	'cultura': {'Precision': 0.83, 'Recall': 0.84, 'F1': 0.84},
	'economía': {'Precision': 0.83, 'Recall': 0.83, 'F1': 0.83},
	'justicia': {'Precision': 0.67, 'Recall': 0.65, 'F1': 0.66},
	'show': {'Precision': 0.97, 'Recall': 0.97, 'F1': 0.97},
	'salud': {'Precision': 0.83, 'Recall': 0.83, 'F1': 0.83},
	'tecnología': {'Precision': 0.84, 'Recall': 0.82, 'F1': 0.83},
	'finanzas': {'Precision': 0.83, 'Recall': 0.80, 'F1': 0.82},
	'gossip': {'Precision': 0.92, 'Recall': 0.82, 'F1': 0.87},
	'crimen': {'Precision': 0.95, 'Recall': 0.92, 'F1': 0.94},
}


lstm = {
	'deportes': {'Precision': 0.98, 'Recall': 0.94, 'F1': 0.96},
	'policiaca': {'Precision': 0.92, 'Recall': 0.94, 'F1': 0.93},
	'espectáculos': {'Precision': 0.95, 'Recall': 0.93, 'F1': 0.94},
	'política': {'Precision': 0.74, 'Recall': 0.81, 'F1': 0.77},
	'seguridad': {'Precision': 0.84, 'Recall': 0.83, 'F1': 0.84},
	'sociedad': {'Precision': 0.61, 'Recall': 0.58, 'F1': 0.60},
	'cultura': {'Precision': 0.78, 'Recall': 0.83, 'F1': 0.81},
	'economía': {'Precision': 0.80, 'Recall': 0.86, 'F1': 0.83},
	'justicia': {'Precision': 0.71, 'Recall': 0.69, 'F1': 0.70},
	'show': {'Precision': 0.91, 'Recall': 0.97, 'F1': 0.94},
	'salud': {'Precision': 0.87, 'Recall': 0.79, 'F1': 0.83},
	'tecnología': {'Precision': 0.84, 'Recall': 0.82, 'F1': 0.83},
	'finanzas': {'Precision': 0.78, 'Recall': 0.81, 'F1': 0.79},
	'gossip': {'Precision': 0.84, 'Recall': 0.78, 'F1': 0.81},
	'crimen': {'Precision': 0.95, 'Recall': 0.91, 'F1': 0.93},
}


#     accuracy                           0.86     90035
#    macro avg       0.83      0.83      0.83     90035
# weighted avg       0.87      0.86      0.87     90035




labels = ['NB', 'SVM', 'LSTM', 'LR']

plotLines( (naive, svm, lstm, lgr), labels, 'Recall' )
plot3Bars( (naive, svm, lstm, lgr), labels, 'Recall' )
