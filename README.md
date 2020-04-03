# spanish-text-classification
Approaches about Spanish text classification task with news documents

## Dataset

The datasets are about Mexican news and have 15 categories.

### Downloads

* [MxCN<sub>o</sub> dataset](https://drive.google.com/file/d/1yag5gLaCbT1GWoxYEY1e2IrmZImfM9oe/view?usp=sharing)
* [MxCN<sub>b</sub> dataset](https://drive.google.com/file/d/1AL6mede_WPDe0P6nN21iQoxjTTM4sFfn/view?usp=sharing)
* [FastText Embeddings](https://drive.google.com/open?id=1moxnW-VSy99mtFExfU5muEubTQ62iQem)

The table below describes the data by each category for Corpus MxCN<sub>o</sub>.

No  |  category  |  docs   |   tokens    |vocabulary
:---|    :---    |    ---: |    ---:     | ---:
1   | sports     | 99,722  |  18,320,854 | 192,717
2   | police     | 81,181  |  13,399,289 | 121,272
3   | spectacle  | 41,296  |  6,553,798  | 135,490
4   | politics   | 36,321  |  8,367,360  | 112,715
5   | security   | 30,687  |  4,570,902  | 83,542
6   | society    | 27,246  |  5,983,335  | 116,259
7   | culture    | 25,231  |  6,943,514  | 169,443
8   | economy    | 22,249  |  4,844,702  | 86,160
9   | justice    | 21,319  |  4,059,425  | 81,981
10  | show       | 16,516  |  3,067,445  | 93,371
11  | health     | 13,211  |  3,089,372  | 90,452
12  | technology | 12,329  |  2,775,993  | 93,750
13  | finance    | 10,824  |  2,739,271  | 73,658
14  | gossip     | 5,576   |  1,624,179  | 72,893
15  | crime      | 5,389   |  902,655    | 38,528


The table below describes the data by each category for Corpus MxCN<sub>b</sub>.

No  |  category  |  docs   |   tokens    |vocabulary
:---|    :---    |    ---: |    ---:     | ---:
1   | sports     | 5,000   | 1,258,391   | 47,621
2   | police     | 5,000   | 1,212,592   | 35,070
3   | spectacle  | 5,000   | 1,060,194   | 50,828
4   | politics   | 5,000   | 1,429,449   | 49,060
5   | security   | 5,000   |   946,347   | 37,456
6   | society    | 5,000   | 1,301,358   | 55,573
7   | culture    | 5,000   | 1,858,546   | 82,396
8   | economy    | 5,000   | 1,324,393   | 46,062
9   | justice    | 5,000   | 1,176,162   | 43,918
10  | show       | 5,000   | 1,056,987   | 53,523
11  | health     | 5,000   | 1,494,661   | 57,860
12  | technology | 5,000   | 1,347,286   | 61,202
13  | finance    | 5,000   | 1,876,915   | 53,604
14  | gossip     | 5,000   | 2,388,975   | 69,411
15  | crime      | 5,000   |   929,444   | 37,107


### Documents histogram

The figure below shows the documents histogram.

![documents histogram](/img/zoom_docs_hist.png)

## Classifiers

We use 4 text classification classifiers about Mexican online news.

* Naïve Bayes (NB)
* Linear Support Vector Machine (SVM)
* Logistic Regression (LR)
* Bidirectional Long Short-Term Memory (BiLSTM)

### Requirements

Pyhton libraries used in this project:

* scikit-learn==0.21.3
* keras==2.3.1
* tensorflow==2.1.0


### Bi-LSTM model

<img src="/img/model.png" width="60%" alt="model">

The figure below shows the loss and accuracy function to train the Bi-LSTM network for Corpus MxCN<sub>o</sub>.

![model](/img/acc_loss.png)

The figure below shows the loss and accuracy function to train the Bi-LSTM network for Corpus MxCN<sub>b</sub>.

![model](/img/acc_loss_5k.png)

## Results

### Classifiers evaluation on the MxCN<sub>o</sub> corpus
The **Precision metric** applied over the fifteen news categories is shown in the below Figure.

<img src="/img/Precision_score_lines_full.png" width="100%" alt="precision score">

The **Recall metric** applied over the fifteen news categories is shown in the below Figure.

<img src="/img/Recall_score_lines_full.png" width="100%" alt="recall score">

The **F1 metric** applied over the fifteen news categories is shown in the below Figure.

<img src="/img/F1_score_lines_full.png" width="100%" alt="F1 score">

### Classifiers evaluation on the MxCN<sub>b</sub> corpus
The **Precision metric** applied over the fifteen news categories is shown in the below Figure.

<img src="/img/Precision_score_lines_5k.png" width="100%" alt="precision score">

The **Recall metric** applied over the fifteen news categories is shown in the below Figure.

<img src="/img/Recall_score_lines_5k.png" width="100%" alt="recall score">

The **F1 metric** applied over the fifteen news categories is shown in the below Figure.

<img src="/img/F1_score_lines_5k.png" width="100%" alt="F1 score">


### General results

The table below describes the general evaluations obtained over the four classifiers.

<table>
<thead>
<tr>
<th rowspan="2">No.</th>
<th rowspan="2">Classifier</th>
<th colspan="4">Corpus MxNC<sub>o</sub></th>
<th colspan="4">Corpus MxNC<sub>b</sub></th>
</tr>
<tr>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
	<td>1</td><td>NB</td><td>0.66</td><td>0.42</td><td>0.41</td><td>0.65</td><td>0.81</td><td>0.81</td><td>0.81</td><td>0.80</td></tr>
<tr>
	<td>2</td><td>SVM</td><td>0.70</td><td>0.49</td><td>0.49</td><td>0.68</td><td>0.79</td><td>0.79</td><td>0.78</td><td>0.78</td></tr>
<tr>
	<td>3</td><td>LR</td><td>0.81</td><td>0.80</td><td>0.80</td><td>0.84</td><td><strong>0.86</strong></td><td><strong>0.86</strong></td><td><strong>0.86</strong></td><td><strong>0.86</strong></td></tr>
<tr>
	<td>4</td><td>BiLSTM</td><td><strong>0.85</strong></td><td><strong>0.85</strong></td><td><strong>0.85</strong></td><td><strong>0.88</strong></td><td><strong>0.86</strong></td><td><strong>0.86</strong></td><td><strong>0.86</strong></td><td><strong>0.86</strong></td></tr>
</tbody>
</table>

