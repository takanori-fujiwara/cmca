import itertools
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import utils

import cca


class CMCA(cca.CCA):
    '''
    Contrastive Multiple Correspondence Analysis (cMCA) from Fujiwara and Liu,
    2020 (https://arxiv.org/abs/2007.04540).
    We referred to MCA in Prince library (https://github.com/MaxHalford/prince)
    for this implementation. We modified the part related to perform
    contrastive analysis.

    Parameters
    ----------
    n_components: int, optional, (default=2)
        A number of componentes to take.
    copy: bool, optional, (default=True)
        If False, data passed to fit are overwritten.
    check_input: bool, optional, (default=True)
        If True, validate input datasets
    Attributes
    ----------
    components: ndarray, shape(n_samples, n_components)
        Components (i.e., projection matrix W_T in the paper) obtained after
        fit() with cMCA.
    categories: list, length n_categories
        All category names across questions produced by applying one-hot
        encoder.
    loadings: ndarray, shape(n_categories, n_components)
        Contrastive principal component loadings obtained after fit() with cMCA.
    alpha: float
        The most recently used alpha. This will be manually selected alpha or
        the best alpha selected automatically.
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> from cmca import CMCA

    >>> # Congressional Voting Records Data Set
    >>> # https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
    >>> df = pd.read_csv('./data/house-votes-84.data', header=None)
    >>> with open('./data/house-votes-84.col_names', 'r') as f:
    ...     # chr(10) is newline (to avoid newline when generating doc with sphinx)
    ...     df.columns = [line.replace(chr(10), '') for line in f]
    >>> X = df.iloc[:, 1:]
    >>> y = np.array(df.iloc[:, 0])

    >>> fg = X.iloc[y == 'democrat']
    >>> bg = X.iloc[y == 'republican']

    >>> ### cMCA
    >>> cmca.fit(fg=fg, bg=bg, alpha=10)

    >>> # row coordinates (cloud of individuals)
    >>> Y_fg_row = np.array(cmca.transform(fg, axis='row'))
    >>> Y_bg_row = np.array(cmca.transform(bg, axis='row'))

    >>> # col coordinates (cloud of categories)
    >>> Y_fg_col = np.array(cmca.transform(fg, axis='col'))
    >>> Y_bg_col = np.array(cmca.transform(bg, axis='col'))

    >>> # cPC loadings
    >>> loadings = cmca.loadings

    >>> # category names
    >>> categories = cmca.categories

    >>> ### Plot the results
    >>> plt.figure(figsize=[8, 8])

    >>> # plot row coordinates
    >>> plt.subplot(2, 2, 1)
    >>> plt.scatter(Y_fg_row[:, 0], Y_fg_row[:, 1], c='b', s=5, label='demo')
    >>> plt.scatter(Y_bg_row[:, 0], Y_bg_row[:, 1], c='r', s=5, label='rep')
    >>> plt.legend(loc='best', shadow=False, scatterpoints=1, fontsize=10)
    >>> plt.title(f'cMCA row coords. alpha = {alpha}')

    >>> # plot col coordinates
    >>> plt.subplot(2, 2, 2)
    >>> plt.scatter(Y_fg_col[:, 0], Y_fg_col[:, 1], c='g', s=5, label='cate')
    >>> for i in range(Y_fg_col.shape[0]):
    ...     plt.text(Y_fg_col[i, 0], Y_fg_col[i, 1], str(i), fontsize=6)
    >>> plt.legend(loc='best', shadow=False, scatterpoints=1, fontsize=10)
    >>> plt.title(f'cMCA col coords. alpha = {alpha}')

    >>> # plot cPC loadings
    >>> plt.subplot(2, 2, 3)
    >>> plt.scatter(loadings[:, 0], loadings[:, 1], c='g', s=5, label='cate')
    >>> for i in range(loadings.shape[0]):
    ...     plt.text(loadings[i, 0], loadings[i, 1], str(i), fontsize=6)
    >>> plt.legend(loc='best', shadow=False, scatterpoints=1, fontsize=10)
    >>> plt.title(f'cPC loadings. alpha = {alpha}')

    >>> # plot category names
    >>> plt.subplot(2, 2, 4)
    >>> for i, cate in enumerate(categories):
    ...     plt.text(0.1, i, str(i) + ': ' + cate, fontsize=6, c='g')
    >>> plt.xticks([])
    >>> plt.yticks([])
    >>> plt.xlim([0, 1])
    >>> plt.ylim([len(categories) + 1, -1])
    >>> plt.title('Categories')
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

    def _trace_ratio(self, fg, bg, eta):
        tr_fg = (
            self.components.transpose() @ self.B_fg @ self.components).trace()

        # this is the way to add eta in cNRL by Fujiwara et al., 2020.
        # https://arxiv.org/abs/2005.12419
        # tr_bg = (self.components.transpose() @ self.B_bg @ self.components +
        #          np.identity(self.components.shape[1]) * eta).trace()

        # here is the new way to add eta to make sure eta is the ratio of tr_fg
        tr_bg = (self.components.transpose() @ self.B_bg
                 @ self.components).trace() + tr_fg * eta

        return tr_fg / tr_bg

    def fit(self,
            fg,
            bg,
            auto_alpha_selection=True,
            alpha=None,
            eta=1e-3,
            convergence_ratio=1e-2,
            max_iter=10,
            y=None):
        """Fit the model with target and background datasets.

        Parameters
        ----------
        fg: pandas dataframe, shape (n_samples, n_questions)
            A target (or foreground) categorical dataset.
        bg: pandas dataframe, shape (n_samples, n_questions)
            A background categorical dataset. The columns of bg must be the same
            with fg. (A row size can be different from fg.)
        auto_alpha_selection:
            If True, find auto_alpha_selection for fit. Otherwise, compute PCs
            based on input alpha.
        alpha: float
            A contrast parameter, which quantifies the trade-off between having
            high target variance and low background variance. alpha must be
            equal to or larger than 0. If 0, the result will be the same with
            the ordinary PCA. If auto_alpha_selection is True, this alpha is
            used as an initial alpha value for auto selection.
        eta: float, optional, (default=1e-3)
            Small constant value that will add to covariance matrix of bg when
            applying automatic alpha selection. eta relates to the maximum
            alpha that will be considered as the best alpha. For example,
            eta=1e-3 allows that alpha reaches till 1e+3.
        convergence_ratio: float, optional, (default=1e-2)
            Threshold of improvement ratio for convergence of automatic alpha
            selection.
        max_iter=10: int, optional, (default=10)
            The number of alpha updates at most.
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports are the history of "alpha" values.
        y: dummy paramter
        Returns
        -------
        self.
        """
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
        if alpha is None:
            alpha = 0
        super().fit(G_fg, G_bg, alpha=alpha, y=y)

        if auto_alpha_selection:
            new_alpha = self._trace_ratio(fg, bg, eta)
            # TODO: this can be faster by saving B_fg and B_tg and only update
            # alpha value when rerunning fit()
            while max_iter > 0 and new_alpha > alpha and (
                    new_alpha - alpha) / (alpha + 1e-15) > convergence_ratio:
                alpha = new_alpha
                super().fit(G_fg, G_bg, alpha=alpha, y=y)
                new_alpha = self._trace_ratio(fg, bg, eta)
                max_iter -= 1

        # Compute the total inertia
        n_new_columns = G_fg.shape[1]
        self.total_inertia_ = (n_new_columns -
                               n_initial_columns) / n_initial_columns

        return self

    def _row_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        G = CMCA.OneHotEncoder(self.cate_each_q_).fit(X).transform(X)

        return super().row_coordinates(G)

    def _col_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        G = CMCA.OneHotEncoder(self.cate_each_q_).fit(X).transform(X)

        return super().col_coordinates(G)

    def transform(self, X, axis='row'):
        """Compute row coordinates (or cloud of individuals) or column
        coordinates (or cloud of categories) with learned components by fit.

        Parameters
        ----------
        X: pandas dataframe, shape (n_samples, n_questions)
            A categorical dataset.
        axis: 'row' or 'col', optional, default='row'
            If 'row', compute row coordniates. Otherwise, compute column
            coordinates.
        Returns
        -------
        coordinates: ndarray, shape(n_samples or n_categories, n_components)
            If axis is 'row', row coordinates of X, shape(n_samples, n_components).
            Otherwise, column coordinates of X, shape(n_categories, n_components).
        """
        utils.validation.check_is_fitted(self)
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self._row_coordinates(
            X) if axis == 'row' else self._col_coordinates(X)
