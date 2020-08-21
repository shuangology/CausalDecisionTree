# CausalDecisionTree
A Causal Decision Tree algorithm for causality inference with continuous variables

This is a set of scripts originally written for one chapter of my thesis on investor preference in CBBC investing.

CBBC (Callable Bull/Bear Contract) is a derivative product traded in HKEx. 
One speciality of this financial product is that it inhibits many contract features. CBBC is generally very unevenly 
traded in HongKong market, with less 10% of the listed products attracting more than 90% of the daily turnover. Apart from 
contract features, (historical) market performance and underlying perfermance could also play a role in determining its attractiveness
to the investors. Therefore, I included 71 features for one CBBC and use a causal decision tree to learn the causality relationship with 
the features one CBBC has and its attractiveness to the investors. 

Causal Decision Tree (CDT) differs from traditional decision tree model in machine learning that each of its branch leads to 
a relationship of causality instead of classification. Output of CDT can have some implications or overlap with the classic decision tree result
but the way how branches split is not necessarily the most efficient way to classify the dataset. Orginal idea of causal decision tree can be seen in 
Li J et al.(2016). There algorithm require input features and out put features to be both binary. I here extend their algorithm into a more general way,

The peseudo code is as the following:

![屏幕快照 2020-08-21 下午6 32 53](https://user-images.githubusercontent.com/43864477/90919424-58c87b00-e3de-11ea-9b13-acdbdc64df25.png)

![屏幕快照 2020-08-21 下午6 57 11](https://user-images.githubusercontent.com/43864477/90920398-2455be80-e3e0-11ea-9ced-d4f955c320e8.png)

![屏幕快照 2020-08-21 下午6 59 13](https://user-images.githubusercontent.com/43864477/90920563-6d0d7780-e3e0-11ea-9d54-56e8da9710da.png)

The algorithm itself could be applied in other data analyse 
I would be happy to discuss if you have any questions regarding my code. Unfortunately, 
my thesis is required to be written in Chinese, so currently I have no English version of detailed description of this algorithm yet. 
I am still working on to put it in English, please be patient for further update.



Reference :

Li J, Ma S, Le T, et al. Causal decision trees[J]. IEEE Transactions on Knowledge and Data Engineering, 2016, 29(2): 257-271.