# CausalDecisionTree
A Causal Decision Tree algorithm for causality inference with continuous variables

This is a set of scripts originally written for one chapter of my thesis on investor preference in CBBC investing.

CBBC(Callable Bull / Bear Contract) is a derivative product traded in HKEx.
One speciality of this financial product is that it inhibits many contract features. CBBC is generally very unevenly
traded in HongKong market, with less 10 % of the listed products attracting more than 90 % of the daily turnover. Apart from
contract features, (historical) market performance and underlying performance could also play a role in determining its attractiveness
to the investors. When one investor is making trading decisions from a variety financial products, the whole process is usually not linear but hierarchy(tree - like) .


In other words, investors make decisions based on a sequence of variables and reach to the final decision based on a hierarchical 'if-else' questions.
Multiple variables are not likely to be proceeded in parallel in human mind. This results in a different inference tools. In classical econometrics, a typical regression approach would be to fit a logit model to a binary variable.
However, even thought the regressors are correct, a logit model may still fail to make good inference. Logit model do not recognize that the data set embeds a hierarchical structure.

For example, when linking lunch options with multiple lunch - related features, if time efficiency is what we currently look at, the first variable would be the distance of the restaurant,
taste, price, food allergy and other features would come afterwards in sequence. A close and fair priced stall attracts higher rate of consumers in busy working days than each of these attributes
taken independently. If picking a potential lunch date place, a different sequence of the variables would be adopted. 

Therefore, I included 71 features for one CBBC and use a causal decision tree to learn the causality relationship with
the features one CBBC has and its attractiveness to the investors.


Causal Decision Tree(CDT) differs from traditional decision tree model in machine learning that each of its branch leads to
a relationship of causality instead of classification. Output of CDT can have some implications or overlap with the classic decision tree result
but the way how branches split is not necessarily the most efficient way to classify the dataset.


Original idea of causal decision tree can be seen in
Li J et al.(2016). There algorithm require input features and out put features to be both binary. I here extend their algorithm into a more general way,

The pseudo code is as the following:

<p align="center">
  <img src="https://user-images.githubusercontent.com/43864477/90919424-58c87b00-e3de-11ea-9b13-acdbdc64df25.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/43864477/90920398-2455be80-e3e0-11ea-9ced-d4f955c320e8.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/43864477/90920563-6d0d7780-e3e0-11ea-9d54-56e8da9710da.png" />
</p>



An output causal decision tree looks like as the following:

<p align="center">
  <img src="https://user-images.githubusercontent.com/43864477/91431874-e4b42a00-e858-11ea-993a-b676d826ff13.png" />
</p>



CausalDT.py is where the Tree Constructing and Tree pruning are implemented.
It read data from data folder and also includes a neat tree print function, producing results in ouputCDT folder.

mypymatch is a package implemented the propensity matching method used in tree branch splitting.


The algorithm itself could be applied in other data analyse and I would be happy to discuss if you have any questions regarding my code. Unfortunately,
my thesis is required to be written in Chinese, so currently I have no English version of detailed description of this algorithm yet.
I am still working on to put it in English, please be patient for further update.


Reference:

Li J, Ma S, Le T, et al. Causal decision trees[J]. IEEE Transactions on Knowledge and Data Engineering, 2016, 29(2): 257 - 271.
