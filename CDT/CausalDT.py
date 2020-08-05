from collections import Counter
import mypymatch.functions as uf
from mypymatch.Matcher import Matcher
import numpy as np
import pandas as pd

from datetime import date, timedelta
import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp

from collections import defaultdict
import os
import re

import sys
import scipy.stats as stats


import sys
sys.path.append(sys.argv[0])
sys.setrecursionlimit(1000000)

from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("error",message = 'Maximum number of iterations has been exceeded', category=ConvergenceWarning)
#######################


def turnover_class(Y):
    """
    Y: Rank_Turn / DayOut
    generate rules converting y to y' ; nominal out come value converted into sub classes:
    heavy sell (100,000 dollar sell) ; sell ; zero ; buy ; heavy buy

    return a array of y' in accordance of y, and a dictionary of outcome and its classification rule in a dictionary
    """
    col = Y.name
    Y = pd.DataFrame(Y)
    Y = Y.reset_index().sort_values(by=col)  # to fast the speed


    if col =='Rank_Turn':
        turnover_threshold = 0.3


        mask_z = (Y[col] == 1.0) # rank 1.0 means turnover = 0
        mask_b = (Y[col] < 1.0) & (Y[col] >= turnover_threshold)
        mask_hb = Y[col] < turnover_threshold

        Y = Y.assign(cls='')
        Y.cls = Y.cls.mask(mask_z, 'Zero')
        Y.cls = Y.cls.mask(mask_b, 'Norm')
        Y.cls = Y.cls.mask(mask_hb, 'Popular')
    elif col =='DayOut':
        turnover_threshold = 50

        mask_hs = Y[col] < -1 * turnover_threshold
        mask_s = (Y[col] < 0) & (Y[col] >= (-1 * turnover_threshold))
        mask_z = Y[col] == 0
        mask_b = (Y[col] > 0) & (Y[col] <= turnover_threshold)
        mask_hb = Y[col] > turnover_threshold

        Y = Y.assign(cls='')
        Y.cls = Y.cls.mask(mask_hs, 'HeavySell')
        Y.cls = Y.cls.mask(mask_s, 'Sell')
        Y.cls = Y.cls.mask(mask_z, 'Zero')
        Y.cls = Y.cls.mask(mask_b, 'Buy')
        Y.cls = Y.cls.mask(mask_hb, 'HeavyBuy')

    return Y


class MyRows:

    def __init__(self, rows, outcome_col):
        self.value = rows
        self.dependent_name = outcome_col
        self.dependent = rows[self.dependent_name]
        self.corr_data = rows

    def get_correlated_features(self, alpha=np.nan):
        """
        Get un-correlated feature rows out from the data sample
        Parameters:
        df: pd.Dataframe, features columns + outcome columns
        outcome_col: object, the column name of outcome
        alpha: float, choice of significant level for t-test to keep the correlated variables.

        ----
        return: df : pd.DataFrame ; correlated features + outcome col
        """
        if np.isnan(alpha):
            global args
            alpha = args.alpha
        df = self.value
        outcome_col = self.dependent_name

        df = pd.get_dummies(df)
        if pd.DataFrame.isna(df).any().any():
            raise ValueError('Input feature dataframe contains NaN.')
        if len(df)<3:
            return df

        # change '-' in the column names into '_'
        df.columns = df.columns.str.strip().str.replace('-', '_')

        # only get numerical columns to check if

        no_col = df.select_dtypes(
            include=[
                'int64',
                'float64']).columns.drop(outcome_col)

        for col in no_col:
            arr = df[col]
            outcome = df[outcome_col]
            corr, pvalue = stats.pearsonr(arr, outcome)
            if pvalue > alpha:
                # if fail to reject the null hypothesis that the correlation
                # coefficient IS NOT significantly different from 0.
                df = df.drop(col, axis=1)  # remove the column
        df = df.reset_index(drop=True)

        self.corr_data = df

        return df


def find_best_question(Rows, question_excluded):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain.
    para: question_excluded, questions already asked during building the tree"""
    Rows.get_correlated_features()
    rows, outcome_col = Rows.corr_data, Rows.dependent_name
    best_pvalue = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it

    question_list = rows.columns.drop(Rows.dependent_name)
    qkeys, qcount = np.unique(question_excluded, return_counts=True)
    qdict = defaultdict(list)
    # maxAskCount = 5 # delete the questions that are asked twice
    #
    # for (c, k) in zip(qcount, qkeys):
    #     qdict[c].append(k)
    # if maxAskCount in qdict.keys():
    #     # if the col is used more than maxAskCount,
    #     # remove from the questionlist
    #     for item in qdict[maxAskCount]:
    #         if item in question_list:
    #             question_list = question_list.drop(item)

    if len(question_list) == 0:
        import warnings
        warnings.warn('Find Best Question: Rows is empty')
        return best_pvalue, best_question


# get the df for processing
    res_df = pd.DataFrame(columns=['col', 'val', 'pvalue','gain'])
    ind = 0

    for col in question_list:  # for each feature

        values = set(rows[col])  # unique values in the column

        if is_numeric(
                list(values)[0]):  # if too many numeric value in ValueSet,deduct some
            global args

            SplitCount = args.sp
            if len(values) > SplitCount:
                values = np.linspace(min(rows[col]), max(
                    rows[col]), SplitCount)[1:-1]

        for val in values:  # for each value
            res_df.loc[ind, :] = [col, val, 0.0,0.0]
            ind += 1

    def cal_pvalue_gain(ind):
        [col, val] = res_df.loc[ind, ['col', 'val']]
        question = Question(col, val)

        # try splitting the dataset
        true_rows, false_rows = partition(rows, question)
        TrRows = MyRows(true_rows, Rows.dependent_name)
        FlRows = MyRows(false_rows, Rows.dependent_name)

        # Skip this split if it doesn't divide the
        # dataset.
        if len(true_rows) == 0 or len(false_rows) == 0:
            return

        try:
            # Get Prospensity_matched dataset

            matchdf = prospensity_match(rows, question, outcome_col)

            # Calculate the p-value from this split
            pvalue = match_ttest(matchdf, question, outcome_col)

        except:

            pvalue = 0.0
        # Calculate the information gain from this split
        current_uncertainty = gini_r(Rows)
        gain = info_gain(TrRows, FlRows, current_uncertainty)

        res_df.loc[ind,'gain'] = gain # lowest the better gini

        res_df.loc[ind, 'pvalue'] = pvalue



    # Start multiprocessing

    cpu_cores = mp.cpu_count()
    if cpu_cores > 10:
        num_cores = int(np.floor(mp.cpu_count() / 4))
    else:
        num_cores = cpu_cores - 1

    Parallel(n_jobs=num_cores)(delayed(cal_pvalue_gain)(i)
                               for i in tqdm(res_df.index, desc='calculating p_value and gini for {} rows'.format(len(Rows.value))))


    weights = [0.15,0.85]
    res_df['ranks']= res_df[['pvalue','gain']].mul(weights).sum(1) / res_df.shape[0] # weighted rank over pvalue and gini info gain
    res_df = res_df.sort_values(by=['ranks'],ascending=False)
    best_row = res_df.head(1)

    best_question = Question(
        best_row['col'].values[0],
        best_row['val'].values[0])

    return best_pvalue, best_question



def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
            return "{0} {1} {2:.4f}" .format(
                self.column, condition, self.value)
        return "%s %s %s" % (
            self.column, condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows = pd.DataFrame(columns=rows.columns)
    false_rows = pd.DataFrame(columns=rows.columns)
    for i in rows.index:
        row = rows.loc[i, :]
        if question.match(row):
            true_rows = true_rows.append(row)
        else:
            false_rows = false_rows.append(row)
    return true_rows, false_rows


def varnameQuestion(question):
    Yvar = question.column
    Value = question.value
    if is_numeric(Value):
        critic = '_leq_'
        val_str = str(np.round(Value, 2)).replace('.', '_')
    else:
        critic = '_is_'
        val_str = Value
    return Yvar + critic + val_str


def prospensity_match(df_rows, question, outcome_col):
    '''
    according to the binary question, return matched df_rows
    '''
    true_rows, false_rows = partition(df_rows, question)

    Yvar = question.column
    Yvar_new = varnameQuestion(question)
    # get the binary attribute value colum as Yvar

    true_rows = true_rows.rename(columns={Yvar: Yvar_new})
    false_rows = false_rows.rename(columns={Yvar: Yvar_new})

    true_rows[Yvar_new] = 1
    false_rows[Yvar_new] = 0

    # Before getting into propensity match, we should exclude cols that cause
    # perfect separations

    categorical_cols = list(
        df_rows.select_dtypes(
            include=[
                'bool',
                'object']).columns)
    if Yvar in categorical_cols:
        categorical_cols.remove(Yvar)  # in case of yvar is an bool or object

    # if only categorical_cols in the df_rows cols, then directly return
    # without matching

    if len(categorical_cols) + 2 == len(df_rows.columns):
        df_rows = df_rows.rename(columns={Yvar: Yvar_new})
        return df_rows

    m = Matcher(
        false_rows,
        true_rows,
        yvar=Yvar_new,
        exclude=categorical_cols +
        [outcome_col])
    # np.random.seed(20170925)
    acc = m.fit_scores(balance=True, ret=True)
    if abs(
            acc -
            0.5) < 0.01:  # if it is already a balanced dataset, then no need for propensity match
        return m.data
    else:
        try:
            m.predict_scores()
        except Exception as e:
            # if error in
            print('Predict Score Error:{}, We adopt random scores here'.format(e))
            m.data['scores'] = np.random.rand(len(m.data))
            return m.data
        m.match(method="min", nmatches=1, threshold=0.0001)
        m.assign_weight_vector()
        return m.matched_data


def match_ttest(matchdf, question, outcome_col):
    classification_var = varnameQuestion(question)
    try:
        X = matchdf[matchdf[classification_var] == 1][outcome_col]
        Y = matchdf[matchdf[classification_var] == 0][outcome_col]
    except BaseException:
        print(matchdf.columns + '+' + classification_var)
    from scipy.stats import ttest_ind
    if len(X) < 2 or len(Y) < 2:
        return 0
    else:
        tstats, pvalue = ttest_ind(X, Y)
        return pvalue


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, Rows):
        res = turnover_class(Rows.dependent)
        keys, count = np.unique(res.cls, return_counts=True)
        self.predictions = dict(zip(keys, count))
        self.value = max(self.predictions, key=self.predictions.get)
        self.real = []



class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch,dependent):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.dependent = dependent
        self.real = []  # if prunned as a leaf, save it here

        res = turnover_class(self.dependent)
        keys, count = np.unique(res.cls, return_counts=True)
        self.predictions = dict(zip(keys, count))




def build_tree(Rows, height, question_excluded=[]):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if height == 0 or len(Rows.value.columns) == 1:
        return Leaf(Rows)

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    pvalue, question = find_best_question(Rows, question_excluded)
    if not question:  # if no questions can be asked, then return a leaf node
        return Leaf(Rows)
    else:
        question_excluded.append(question.column)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    Rows.get_correlated_features()
    true_rows, false_rows = partition(Rows.corr_data, question)
    if len(true_rows) == 0 or len(false_rows) == 0:
        return Leaf(Rows)
    elif len(false_rows) == 0 and len(true_rows) == 0:
        raise ValueError('Empty rows from partition')
    else:
        TrRows = MyRows(true_rows, Rows.dependent_name)
        FlRows = MyRows(false_rows, Rows.dependent_name)

    # Recursively build the true branch.
    # CurrentNodePos = LastNodePos+1
    true_branch = build_tree(TrRows, height - 1, question_excluded)

    # Recursively build the false branch.

    false_branch = build_tree(FlRows, height - 1, question_excluded)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # the last node,
    # as well as the branches to follow
    # depending on the answer.
    return Decision_Node(question, true_branch, false_branch,Rows.dependent)


# BEST PARTITION
def gini(counts):
    """Calculate the Gini Impurity for a list of counts.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    impurity = 1
    sum_count = sum([val for val in counts.values()])
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(sum_count)
        impurity -= prob_of_lbl ** 2
    return impurity

def gini_r(Rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """

    counts = Leaf(Rows).predictions

    return gini(counts)

def info_gain(leftRows, rightRows, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    left = leftRows.value
    right = rightRows.value
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini_r(leftRows) - (1 - p) * gini_r(rightRows)



def print_tree(
    node,
    height=0,
    spacing="",
    sourceFile=open(
        '../mytree.txt',
        'a+'),last_node = 0,branch_type = ''):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        this_node = generate_ind()
        print(str(last_node)+' -> '+str(this_node)+' [headlabel='+branch_type+'] ;', file=sourceFile)
        print(str(this_node)+ ' [label="'+spacing+ spacing + "\\n", node.value, "\\n",node.predictions,
              " \\nGINI:{0:.2f}".format(gini(node.predictions)),'"] ;', file=sourceFile)
        return
    if isinstance(node.real, Leaf):
        this_node = generate_ind()
        print(str(last_node) + ' -> ' + str(this_node)+' [headlabel='+branch_type+'] ;', file=sourceFile)
        print(str(this_node) + ' [label="' + spacing  + spacing + "\\n", node.real.value,"\\n",
              node.real.predictions,
              " \\nGINI:{0:.2f}".format(gini(node.predictions)), '"] ;', file=sourceFile)
        return

    # Print the question at this node
    this_node = generate_ind()
    if last_node!=0:
        print(str(last_node) + ' -> ' + str(this_node) +' [headlabel='+branch_type+'] ;', file=sourceFile)
    print(str(this_node)+ ' [label="'+spacing + str(node.question)+spacing + "\\n", node.predictions,
              " \\nGINI:{0:.2f}".format(gini(node.predictions)),'"] ;', file=sourceFile)

    # Call this function recursively on the true branch
    #print(str(height) + spacing + '--> True:', file=sourceFile)
    print_tree(node.true_branch, height + 1, spacing + "  ", sourceFile,this_node,'True')

    # Call this function recursively on the false branch
    #print(str(height) + spacing + '--> False:', file=sourceFile)
    print_tree(node.false_branch, height + 1, spacing + "  ", sourceFile,this_node,'False')

node_num = 0
def generate_ind():
    global node_num
    node_num = node_num+1
    return node_num

def TreePruning(node):
    if isinstance(node, Leaf):
        return
    if isinstance(node.real, Leaf):
        return

    original_leaf_check = isinstance(
        node.true_branch,
        Leaf) and isinstance(
        node.false_branch,
        Leaf)  # if leaf as both children
    real_leaf_check = isinstance(  # if both children has already been pruned as leaf
        node.true_branch.real,
        Leaf) and isinstance(
        node.false_branch.real,
        Leaf)
    if original_leaf_check:
        if node.true_branch.value == node.false_branch.value:
            node.real = node.true_branch
            node.real.predictions = dict(
                Counter(
                    node.true_branch.predictions) +
                Counter(
                    node.false_branch.predictions))
            return
    if real_leaf_check:
        if node.true_branch.real.value == node.false_branch.real.value:
            node.real = node.true_branch.real
            node.real.predictions = dict(
                Counter(
                    node.true_branch.real.predictions) +
                Counter(
                    node.false_branch.real.predictions))
            return
        # if both child are the same, then delete the leaves, turn the father
        # node into a leaf

    TreePruning(node.true_branch)
    TreePruning(node.false_branch)


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        description='A script for causal decision tree for continuous varible')
    parser.add_argument(
        "--sp",
        default=6,
        type=int,
        help="Number of split for continuous  ")
    parser.add_argument(
        "--hmax",
        default=5,
        type=int,
        help="Maximum height of the tree")
    parser.add_argument(
        "--random",
        default=False,
        type=bool,
        help="Whether we random pick samples from original data (for testing)")
    parser.add_argument(
        "--pick",
        default=2000,
        type=int,
        help="Number of random pick from the original data")
    parser.add_argument(
        "--alpha",
        default=0.1,
        type=float,
        help="Significance level for correlation check")
    parser.add_argument(
        "--dep",
        default = 'Rank_Turn',
        type = str,
        help = 'Dependent value of the tree: Rank_Turn: turnover daily ranking ; DayOut: Day change of out contracts '
    )

    global args
    args = parser.parse_args()
    h_max = args.hmax
    random_flag = args.random
    random_pick = args.pick

    import zipfile as zipfile

    info_file_name = '../../data/CBBC_Monthly_Info/TOTAL_HSI_INFO_Update.msg.zip'
    archive = zipfile.ZipFile(info_file_name, 'r')
    hsi_df = pd.read_msgpack(
        archive.open('TOTAL_HSI_INFO_Update.msg'),
        encoding='utf-8')

    # only consider underlying HSI

    hsi_df = hsi_df.dropna()


    if random_flag:
        import random
        random.seed(2021)
        target = hsi_df.loc[random.choices(
            list(hsi_df.index), k=random_pick), :]
    else:
        target = hsi_df

    print('Size of the sample{0},Dependent variable:{1}'.format(len(target),args.dep))

    target['RelaDisToK'] = np.abs(target['Underlying_Close']-target['Strike_Level'])/target['Underlying_Close']

    Star_Issuer = ['CS', 'HT', 'JP', 'SG', 'UB']

    target['StarIssuer'] = target['Issuer'].isin(Star_Issuer).apply(str)

    target = target.rename(columns={'Ratio_of_issue_still_out_in_market_': 'OutRatio', 'Rela_Dis_to_Call': 'RelaDisToC'})

    target = target.sort_values(by = ['UNIQUE','Trade_Date'])

    target.loc[:, 'Theo_Lev'] = [p / (x * r) for x, r, p in
                                   zip(target.Closing_Price, target.loc[:, 'Ent_Ratio'], target.Underlying_Close)]


    if args.dep=='Rank_Turn':
        target = target.dropna()
        target = target.sort_values(by  =['UNIQUE','Trade_Date'])
        target['Last_Rank_Out'] =target.groupby("UNIQUE")['Rank_Out'].shift(1)
        target['Last_Rank_Turn'] = target.groupby("UNIQUE")['Rank_Turn'].shift(1)
        target['Last_DayOut'] = target.groupby("UNIQUE")['DayOut'].shift(1)
        # our target is to see the investor decision based on last trading day stats
        df_set = target[['RelaDisToC',
                         'RelaDisToK',
                         'TimeToMaturity',
                         'Bull_Bear',
                         'StarIssuer',
                         'AQC',
                         'LastTurnover',
                         'FinCost',  #  (Ratio*P - (UnderlyP-Strike)*BB)/Strike
                         'OutRatio',
                         'LastDayReturn',
                         'Theo_Lev',
                         'Last_Rank_Out', 'Last_Rank_Turn','Last_DayOut','Rank_Turn']].dropna()


        df = MyRows(df_set, 'Rank_Turn')
        mytree = build_tree(df, h_max)
    elif args.dep=='DayOut':
        target = target.dropna()
        df_set = target[['RelaDisToC',
                         'RelaDisToK',
                         'TimeToMaturity',
                         'Bull_Bear',
                         'StarIssuer',
                         'AQC',
                         'LastTurnover',
                         'FinCost',  #  (Ratio*P - (UnderlyP-Strike)*BB)/Strike
                         'OutRatio',
                         'LastDayReturn',
                         'Theo_Lev',
                         'Rank_Out', 'Rank_Turn','DayOut']].dropna()
        df = MyRows(df_set, 'DayOut')
        mytree = build_tree(df, h_max)


    else:
        raise ValueError('Not implemented dependent name')

    for i in range(h_max):
        TreePruning(mytree)

    import time
    time.sleep(2)
    var_num = len(df_set.columns)
    SampleSize = len(df_set)
    filename = '../data/CDT/'+args.dep+'_tree' + '_sp_' + \
        str(args.sp) + '_h_' + str(args.hmax) +'_r_' + str(args.random) + '_var_'+str(var_num)+'N'+str(SampleSize)
    PrintTreeFile = open(filename+'.txt', 'w')
    print('Printing Tree on ', filename)
    print('digraph Tree {\nnode [shape=box] ;\n',file=PrintTreeFile)
    print_tree(mytree, sourceFile=PrintTreeFile)
    print('}',file=PrintTreeFile)
    PrintTreeFile.close()


    # Plot Tree

    with open(filename+'.txt', "r") as myfile:
        my_dot_data = myfile.read()
    # Draw graph
    import pydotplus
    graph = pydotplus.graph_from_dot_data(my_dot_data)
    # Save graph
    graph.write_pdf(filename+'.pdf')


if __name__ == '__main__':

    _main()
