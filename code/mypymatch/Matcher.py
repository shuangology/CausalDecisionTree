from __future__ import print_function


from . import *
from . import functions as uf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("error",message = 'Maximum number of iterations has been exceeded', category=ConvergenceWarning)


class Matcher:
    """
    Matcher Class -- Match data for an observational study.

    Parameters
    ----------
    test : pd.DataFrame
        Data representing the test group
    control : (pd.DataFrame)
        Data representing the control group
    formula : str (optional)
        custom formula to use for logistic regression
        i.e. "Y ~ x1 + x2 + ..."
    yvar : str (optional)
        Name of dependent variable (the treatment)
    exclude : list  (optional)
        List of variables to ignore in regression/matching.
        Useful for unique idenifiers
    """

    def __init__(self, test, control, yvar, formula=None, exclude=[]):
        # configure plots for ipynb
        plt.rcParams["figure.figsize"] = (10, 5)
        # variables generated during matching
        aux_match = ['scores', 'match_id', 'weight', 'record_id']
        # assign unique indices to test and control
        t, c = [i.copy().reset_index(drop=True) for i in (test, control)]
        t = t.dropna(axis=1, how="all")
        c = c.dropna(axis=1, how="all")
        c.index += len(t)
        self.data = t.dropna(
            axis=1,
            how='all').append(
            c.dropna(
                axis=1,
                how='all'),
            sort=True)
        self.control_color = "#1F77B4"
        self.test_color = "#FF7F0E"
        self.yvar = yvar
        self.exclude = exclude + [self.yvar] + aux_match
        self.formula = formula
        self.nmodels = 1  # for now
        self.models = []
        self.swdata = None
        self.model_accuracy = []
        self.data[yvar] = self.data[yvar].astype(int)  # should be binary 0, 1
        self.xvars = [
            i for i in self.data.columns if i not in self.exclude and i != yvar]
        self.data = self.data.dropna(subset=self.xvars)
        self.matched_data = []
        self.xvars_escaped = ["Q('{}')".format(x) for x in self.xvars]
        self.yvar_escaped = "Q('{}')".format(self.yvar)
        self.y, self.X = patsy.dmatrices('{} ~ {}'.format(
            self.yvar_escaped, '+'.join(self.xvars_escaped)), data=self.data, return_type='dataframe')
        self.xvars = [i for i in self.data.columns if i not in self.exclude]
        self.test = self.data[self.data[yvar]== True]
        self.control = self.data[self.data[yvar] == False]
        self.testn = len(self.test)
        self.controln = len(self.control)
        self.minority, self.majority = [i[1] for i in sorted(
            zip([self.testn, self.controln], [1, 0]), key=lambda x: x[0])]
        # print('Formula:\n{} ~ {}'.format(yvar, '+'.join(self.xvars)))
        # print('n majority:', len(self.data[self.data[yvar] == self.majority]))
        # print('n minority:', len(self.data[self.data[yvar] == self.minority]))

    def fit_scores(self, balance=True, nmodels=None, ret=False):
        """
        Fits logistic regression model(s) used for
        generating propensity scores

        Parameters
        ----------
        balance : bool
            Should balanced datasets be used?
            (n_control == n_test)
        nmodels : int
            How many models should be fit?
            Score becomes the average of the <nmodels> models if nmodels > 1
        ret: Bool


        Returns
        -------
        None
        """
        # reset models if refitting
        if len(self.models) > 0:
            self.models = []
        if len(self.model_accuracy) > 0:
            self.model_accuracy = []
        if not self.formula:
            # use all columns in the model
            self.xvars_escaped = ["Q('{}')".format(x) for x in self.xvars]
            self.yvar_escaped = "Q('{}')".format(self.yvar)
            self.formula = '{} ~ {}'.format(
                self.yvar_escaped, '+'.join(self.xvars_escaped))
        if balance:
            if nmodels is None:
                # fit multiple models based on imbalance severity (rounded up
                # to nearest tenth)
                minor, major = [self.data[self.data[self.yvar] == i]
                                for i in (self.minority, self.majority)]
                nmodels = int(np.ceil((len(major) / len(minor)) / 10) * 10)
            self.nmodels = nmodels
            i = 0
            errors = 0
            while i < nmodels and errors < 3:
                #uf.progress(i+1, nmodels, prestr="Fitting Models on Balanced Samples")
                # sample from majority to create balance dataset
                #df = self.balanced_sample()

                df = self.balance_drop_static_cols()  # so in this function, formula will be changed

                # df = pd.concat([uf.drop_static_cols(df[df[self.yvar] == 1], yvar=self.yvar),
                #                uf.drop_static_cols(df[df[self.yvar] == 0], yvar=self.yvar)],
                #               sort=True)
                # if dropped any static cols formula should be reconstructed

                if self.formula[-2:]=='~ ' : # if all the X columns are droped
                    errors = 3 # force exit the while loop
                    break

                y_samp, X_samp = patsy.dmatrices(
                    self.formula, data=df, return_type='dataframe')
                X_samp.drop(self.yvar, axis=1, errors='ignore', inplace=True)
                if len(np.unique(y_samp)) == 1:
                    raise ValueError('Complete separation on yvar')



                try:
                    glm = sm.Logit(y_samp, X_samp)
                    res = glm.fit()
                except Exception as e:# in case of PerfectSeparationError, use OLS as proxy
                    #if ('Perfect separation' not in str(e)) and ('Singular matrix' not in str(e)):
                        #print('ERROR in GLM :{}'.format(e))

                    try:
                        ols = sm.OLS(y_samp, X_samp)
                        res = ols.fit()
                    except Exception as e:
                        print('ERROR in OLS '.format(e))

                try:
                    self.model_accuracy.append(
                        self._scores_to_accuracy(
                            res, X_samp, y_samp))
                    self.models.append(res)
                    i = i + 1
                except Exception as e:
                    # Complete Separation among data set can cause the maximum
                    # likelihood estimate does not exist
                    errors = errors + 1  # to avoid infinite loop for misspecified matrix
                    print('Error: {}, tried times {}/3\n'.format(e, errors))
            # print("\nAverage Accuracy:", "{}%".
            #     format(round(np.mean(self.model_accuracy) * 100, 2)))
            if self.model_accuracy == []:
                # if cause infinite loop,(linearly separable data set) then
                # return a 0.5 accuracy of the model
                accuracy = 0.5
            else:
                accuracy = np.mean(self.model_accuracy) * 100
        else:
            # ignore any imbalance and fit one model
            print('Fitting 1 (Unbalanced) Model...')
            y_samp, X_samp = patsy.dmatrices(
                self.formula, data=df, return_type='dataframe')
            X_samp.drop(self.yvar, axis=1, errors='ignore', inplace=True)
            try:
                glm = sm.Logit(y_samp, X_samp)
                res = glm.fit()
            except Exception as e:  # in case of PerfectSeparationError, use OLS as proxy
                #if ('Perfect separation' not in str(e)) and ('Singular matrix' not in str(e)):
                    #print('ERROR in GLM :{}'.format(e))
                try:
                    ols = sm.OLS(y_samp, X_samp)
                    res = ols.fit()
                except Exception as e:
                    print('ERROR in OLS :{}'.format(e))
            try:
                self.model_accuracy.append(
                    self._scores_to_accuracy(
                        res, self.X, self.y))
                self.models.append(res)
            except Exception as e:
                # Complete Separation among data set can cause the maximum
                # likelihood estimate does not exist
                print('Error: {}'.format(e))
                return 0.5
            #print("\nAccuracy", round(np.mean(self.model_accuracy[0]) * 100, 2))
            accuracy = np.mean(self.model_accuracy[0]) * 100
        if ret:
            return accuracy

    def predict_scores(self):
        """
        Predict Propensity scores for each observation.
        Adds a "scores" columns to self.data

        Returns
        -------
        None
        """
        scores = np.zeros(len(self.data))
        for i in range(self.nmodels):
            m = self.models[i]
            data_source = patsy.dmatrix('{}'.format('+'.join(self.xvars_escaped + [self.yvar_escaped])),
                                        data=self.data, return_type='dataframe')
            scores += m.predict(data_source[m.params.index])
        self.data['scores'] = scores / self.nmodels

    def match(self, threshold=0.001, nmatches=1, method='min', max_rand=10):
        """
        Finds suitable match(es) for each record in the minority
        dataset, if one exists. Records are exlcuded from the final
        matched dataset if there are no suitable matches.

        self.matched_data contains the matched dataset once this
        method is called

        Parameters
        ----------
        threshold : float
            threshold for fuzzy matching matching
            i.e. |score_x - score_y| >= theshold
        nmatches : int
            How majority profiles should be matched
            (at most) to minority profiles. 1 means one to one match
        method : str
            Strategy for when multiple majority profiles
            are suitable matches for a single minority profile
            "random" - choose randomly (fast, good for testing)
            "min" - choose the profile with the closest score
        max_rand : int
            max number of profiles to consider when using random tie-breaks

        Returns
        -------
        None
        """
        if 'scores' not in self.data.columns:
            print("Propensity Scores have not been calculated. Using defaults...")
            self.fit_scores()
            self.predict_scores()
        test_scores = self.data[self.data[self.yvar] == True][['scores']]
        ctrl_scores = self.data[self.data[self.yvar] == False][['scores']]
        result, match_ids = [], []
        for i in range(len(test_scores)):
            # uf.progress(i+1, len(test_scores), 'Matching Control to Test...')
            match_id = i
            score = test_scores.iloc[i]
            if method == 'random':
                bool_match = abs(ctrl_scores - score) <= threshold
                matches = ctrl_scores.loc[bool_match[bool_match.scores].index]
            elif method == 'min':
                matches = abs(ctrl_scores -
                              score).sort_values('scores').head(nmatches)
            else:
                raise(
                    AssertionError,
                    "Invalid method parameter, use ('random', 'min')")
            if len(matches) == 0:
                continue
            # randomly choose nmatches indices, if len(matches) > nmatches
            select = nmatches if method != 'random' else np.random.choice(
                range(1, max_rand + 1), 1)
            chosen = np.random.choice(
                matches.index, min(
                    select, nmatches), replace=False)
            result.extend([test_scores.index[i]] + list(chosen))
            match_ids.extend([i] * (len(chosen) + 1))
        self.matched_data = self.data.loc[result]
        self.matched_data['match_id'] = match_ids
        self.matched_data['record_id'] = self.matched_data.index

    def select_from_design(self, cols):
        d = pd.DataFrame()
        for c in cols:
            d = pd.concat(
                [d, self.X.select(lambda x: x.startswith(c), axis=1)], axis=1, sort=True)
        return d

    def balanced_sample(self, data=None):
        if not data:
            data = self.data
        minor, major = data[data[self.yvar] == self.minority], \
            data[data[self.yvar] == self.majority]
        return major.sample(len(minor)).append(minor, sort=True).dropna() # random sample from major

    def balance_drop_static_cols(self):
        yvar = self.yvar
        df = self.balanced_sample()
        cols = list(df.columns)
        # will be static for both groups
        cols.pop(cols.index(yvar))
        for col in df[cols]:
            n_unique = len(np.unique(df[col]))
            if n_unique == 1:
                df.drop(col, axis=1, inplace=True)
#                sys.stdout.write('\rStatic column dropped: {}'.format(col))

        current_xvar_escaped = ["Q('{}')".format(
            i) for i in df.columns if i not in self.exclude and i != yvar]
        self.yvar_escaped = "Q('{}')".format(self.yvar)
        self.formula = '{} ~ {}'.format(
            self.yvar_escaped, '+'.join(current_xvar_escaped))
        return df


    def prop_test(self, col):
        """
        Performs a Chi-Square test of independence on <col>
        See stats.chi2_contingency()

        Parameters
        ----------
        col : str
            Name of column on which the test should be performed

        Returns
        ______
        dict
            {'var': <col>,
             'before': <pvalue before matching>,
             'after': <pvalue after matching>}


        """
        if not uf.is_continuous(col, self.X) and col not in self.exclude:
            pval_before = round(
                stats.chi2_contingency(
                    self.prep_prop_test(
                        self.data, col))[1], 6)
            pval_after = round(
                stats.chi2_contingency(
                    self.prep_prop_test(
                        self.matched_data, col))[1], 6)
            return {'var': col, 'before': pval_before, 'after': pval_after}
        else:
            print("{} is a continuous variable".format(col))


    def compare_categorical(self, return_table=False):
        """
        Plots the proportional differences of each enumerated
        discete column for test and control.
        i.e. <prop_test_that_have_x>  - <prop_control_that_have_x>
        Each chart title contains the results from a
        Chi-Square Test of Independence before and after
        matching.
        See mypymatch.prop_test()

        Parameters
        ----------
        return_table : bool
            Should the function return a table with
            test results?

        Return
        ------
        pd.DataFrame() (optional)
            Table with the p-values of the Chi-Square contingency test
            for each discrete column before and after matching

        """
        def prep_plot(data, var, colname):
            t, c = data[data[self.yvar] == 1], data[data[self.yvar] == 0]
            # dummy var for counting
            dummy = [i for i in t.columns if i not in
                     (var, "match_id", "record_id", "weight")][0]
            countt = t[[var, dummy]].groupby(var).count() / len(t)
            countc = c[[var, dummy]].groupby(var).count() / len(c)
            ret = (countt - countc).dropna()
            ret.columns = [colname]
            return ret

        title_str = '''
        Proportional Difference (test-control) for {} Before and After Matching
        Chi-Square Test for Independence p-value before | after:
        {} | {}
        '''
        test_results = []
        for col in self.matched_data.columns:
            if not uf.is_continuous(col, self.X) and col not in self.exclude:
                dbefore = prep_plot(self.data, col, colname="before")
                dafter = prep_plot(self.matched_data, col, colname="after")
                df = dbefore.join(dafter)
                test_results_i = self.prop_test(col)
                test_results.append(test_results_i)

                # plotting
                df.plot.bar(alpha=.8)
                plt.title(title_str.format(col, test_results_i["before"],
                                           test_results_i["after"]))
                lim = max(.09, abs(df).max().max()) + .01
                plt.ylim((-lim, lim))
        return pd.DataFrame(test_results)[
            ['var', 'before', 'after']] if return_table else None

    def prep_prop_test(self, data, var):
        """
        Helper method for running chi-square contingency tests

        Balances the counts of discrete variables with our groups
        so that missing levels are replaced with 0.
        i.e. if the test group has no records with x as a field
        for a given column, make sure the count for x is 0
        and not missing.

        Parameters
        ----------
        data : pd.DataFrame()
            Data to use for counting
        var : str
            Column to use within data

        Returns
        -------
        list
            A table (list of lists) of counts for all enumerated field within <var>
            for test and control groups.
        """
        counts = data.groupby([var, self.yvar]).count().reset_index()
        table = []
        for t in (0, 1):
            os_counts = counts[counts[self.yvar] == t]\
                .sort_values(var)
            cdict = {}
            for row in os_counts.iterrows():
                row = row[1]
                cdict[row[var]] = row[2]
            table.append(cdict)
        # fill empty keys as 0
        all_keys = set(chain.from_iterable(table))
        for d in table:
            d.update((k, 0) for k in all_keys if k not in d)
        ctable = [[i[k] for k in sorted(all_keys)] for i in table]
        return ctable

    def prop_retained(self):
        """
        Returns the proportion of data retained after matching
        """
        return len(self.matched_data[self.matched_data[self.yvar] == self.minority]
                   ) * 1.0 / len(self.data[self.data[self.yvar] == self.minority])



    def record_frequency(self):
        """
        Calculates the frequency of specific records in
        the matched dataset

        Returns
        -------
        pd.DataFrame()
        pd.DataFrame()
            Frequency table of the number records
            matched once, twice, ..., etc.
        """
        freqs = self.matched_data.groupby("record_id") .count().groupby(
            "match_id").count()[["scores"]].reset_index()
        freqs.columns = ["freq", "n_records"]
        return freqs

    def assign_weight_vector(self):
        record_freqs = self.matched_data.groupby("record_id")\
                           .count()[['match_id']].reset_index()
        record_freqs.columns = ["record_id", "weight"]
        fm = record_freqs.merge(self.matched_data, on="record_id")
        fm['weight'] = 1 / fm['weight']
        self.matched_data = fm

    @staticmethod
    def _scores_to_accuracy(m, X, y):
        preds = [[1.0 if i >= .5 else 0.0 for i in m.predict(X)]]
        return (y == preds).sum() * 1.0 / len(y)
