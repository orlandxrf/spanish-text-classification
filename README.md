# spanish-text-classification
Approaches about Spanish text classification task with news documents

## Dataset

The table below describes the data by each document.

No  |  category  |  docs   | vocabulary
:---|    :---    |    ---: |    ---:
1   | sports     | 100,085 | 192,717
2   | police     | 81,489  | 121,272
3   | spectacle  | 41,417  | 135,490
4   | politics   | 36,341  | 112,715
5   | security   | 30,751  | 83,542
6   | society    | 27,255  | 116,259
7   | culture    | 25,239  | 169,443
8   | economy    | 22,349  | 86,160
9   | justice    | 21,322  | 81,981
10  | show       | 16,516  | 93,371
11  | health     | 13,215  | 90,452
12  | technology | 12,336  | 93,750
13  | finance    | 10,872  | 73,658
14  | gossip     | 5,576   | 72,893
15  | crime      | 5,411   | 38,528


The figure below shows the dataset info.

![dataset description](/img/categories.png)

The total sentences number by category as shown in the table below

No  | category   | original  | duplicates | total
:---|   :---     |      ---: |      ---:  |    ---:
1   | sports     | 1,573,920 | 605,859    | 968,061
2   | police     | 1,044,618 | 462,191    | 582,427
3   | culture    | 566,607   | 178,813    | 387,794
4   | politics   | 596,364   | 236,043    | 360,321
5   | spectacle  | 626,054   | 268,887    | 357,167
6   | economy    | 349,232   | 90,770     | 258,462
7   | security   | 329,785   | 142,001    | 187,784
8   | health     | 248,575   | 75,445     | 173,130
9   | justice    | 266,634   | 93,592     | 173,042
10  | society    | 378,346   | 209,902    | 168,444
11  | show       | 234,532   | 77,047     | 157,485
12  | technology | 235,655   | 87,567     | 148,088
13  | finance    | 270,775   | 150,334    | 120,441
14  | gossip     | 136,100   | 68,368     | 67,732
15  | crime      | 69,477    | 24,854     | 44,623
 | | | 6,926,674 | 2,771,673  | 4,155,001

