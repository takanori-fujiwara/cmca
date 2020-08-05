import itertools
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import utils

import cca


class CMCA(cca.CCA):
    '''
    Contrastive Multiple Correspondence Analysis (cMCA). We referred to MCA
    in Prince library (https://github.com/MaxHalford/prince) for this
    implementation. We modified the part related to perform contrastive analysis.
    '''
    class OneHotEncoder(preprocessing.OneHotEncoder):
        '''
        This class is a minor updated version of the customized one-hot encoder
        previously included in Prince library (https://github.com/MaxHalford/prince).
        Because target and background datasets could have a different set of
        categorical values, we cannot use categories='auto'.
        '''
        def __init__(self, categories):
            super().__init__(sparse=True,
                             dtype=np.uint8,
                             categories=categories)
            self.column_names_ = None

        def fit(self, X, y=None):
            self = super().fit(X)
            self.column_names_ = list(
                itertools.chain(
                    *
                    [['{}_{}'.format(col, cat) for cat in self.categories_[i]]
                     for i, col in enumerate(X.columns)]))

            return self

        def transform(self, X):
            oh = pd.DataFrame.sparse.from_spmatrix(super().transform(X))
            oh.columns = self.column_names_
            if isinstance(X, pd.DataFrame):
                oh.index = X.index

            return oh

    def fit(self, fg, bg, alpha, y=None):

        if self.check_input:
            utils.check_array(fg, dtype=[str, np.number])
            utils.check_array(bg, dtype=[str, np.number])

        if not isinstance(fg, pd.DataFrame):
            fg = pd.DataFrame(fg)
        if not isinstance(bg, pd.DataFrame):
            bg = pd.DataFrame(bg)

        n_initial_columns = fg.shape[1]

        # Prince removed one-hot encoder but for our case we need it because we
        # use two different datasets which pottentially have different
        # categories (e.g., fg has y, n for Q1 but bg has only n for Q1)

        # One-hot encode the data to produce disjunctive matrices
        # get categories for each question
        self.cate_each_q_ = [
            np.unique(
                np.concatenate((fg[col_name].unique(), bg[col_name].unique())))
            for col_name in fg.columns
        ]
        # apply one-hot encoder
        G_fg = CMCA.OneHotEncoder(self.cate_each_q_).fit(fg).transform(fg)
        G_bg = CMCA.OneHotEncoder(self.cate_each_q_).fit(bg).transform(bg)

        # Apply CA to the disjunctive matrices
        super().fit(G_fg, G_bg, alpha=alpha, y=y)

        # Compute the total inertia
        n_new_columns = G_fg.shape[1]
        self.total_inertia_ = (n_new_columns -
                               n_initial_columns) / n_initial_columns

        return self

    def row_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        G = CMCA.OneHotEncoder(self.cate_each_q_).fit(X).transform(X)

        return super().row_coordinates(G)

    def col_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        G = CMCA.OneHotEncoder(self.cate_each_q_).fit(X).transform(X)

        return super().col_coordinates(G)

    def transform(self, X, axis='row'):
        """Computes the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self)
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self.row_coordinates(
            X) if axis == 'row' else self.col_coordinates(X)
