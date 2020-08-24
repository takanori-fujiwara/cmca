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
    contrastive analysis. Also, we improved the performance.

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
    >>> cmca.fit(fg, bg)
    >>> # if you set alpha value manually, you can use:
    >>> # cmca.fit(fg, bg, auto_alpha_selection=False, alpha=5)

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

    >>> # After the first fit, if you only update the result with a new alpha,
    >>> # you can use update_fit()
    >>> cmca.update_fit(alpha=1000)
    '''
    class OneHotEncoder(preprocessing.OneHotEncoder):
        '''
        This class is a updated version of the customized one-hot encoder
        previously included in Prince library (https://github.com/MaxHalford/prince).
        Because target and background datasets could have a different set of
        categorical values, in that case, we cannot use categories='auto'.
        Parameters
        ----------
        categories: 'auto' or a list of array-like, optional (default='auto')
            If 'auto', determine categories automatically from the training
            data. Otherwise, categories[i] holds the categories expected in the
            ith column. The passed categories should not mix strings and
            numeric values within a single feature, and should be sorted in
            case of numeric values.
        Attributes
        ----------
        categories_: list of arrays
            The categories of each feature determined during fitting (in order
            of the features in X and corresponding with the output of
            transform). This includes the category specified in drop (if any).
        Examples
        ----------
        >>> from cmca import CMCA

        >>> # Congressional Voting Records Data Set
        >>> # https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
        >>> df = pd.read_csv('./data/house-votes-84.data', header=None)
        >>> with open('./data/house-votes-84.col_names', 'r') as f:
        ...     # chr(10) is newline (to avoid newline when generating doc with sphinx)
        ...     df.columns = [line.replace(chr(10), '') for line in f]
        >>> X = df.iloc[:, 1:]

        >>> X_oh = CMCA.OneHotEncoder().fit_transform(X)

        >>> print(X_oh)
        '''
        def __init__(self, categories='auto'):
            super().__init__(sparse=True,
                             dtype=np.uint8,
                             categories=categories)
            self.column_names_ = None

        def fit_transform(self, X, y=None):
            '''
            Fit OneHotEncoder to X and then transform X.

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                The data to determine the categories of each feature.
            y: None
                Ignored.
            Returns
            ----------
            oh: pandas DataFrame consisting of SparseArray values.
                Transformed input.
            '''
            return self.fit(X, y=y).transform(X)

        def fit(self, X, y=None):
            '''
            Fit OneHotEncoder to X.

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                The data to determine the categories of each feature.
            y: None
                Ignored.
            Returns
            ----------
            self
            '''
            self = super().fit(X)
            self.column_names_ = list(
                itertools.chain(
                    *
                    [['{}_{}'.format(col, cat) for cat in self.categories_[i]]
                     for i, col in enumerate(X.columns)]))

            return self

        def transform(self, X):
            '''
            Transform X using one-hot encoding.

            Parameters
            ----------
            X: array-like, shape(n_samples, n_features)
                The data to determine the categories of each feature.
            Returns
            ----------
            oh: pandas DataFrame consisting of SparseArray values.
                Transformed input.
            '''
            # TODO: This part seems slow. Avoid using pandas dataframe by
            # spearating column names as a different variable
            oh = pd.DataFrame.sparse.from_spmatrix(super().transform(X))
            oh.columns = self.column_names_
            if isinstance(X, pd.DataFrame):
                oh.index = X.index

            return oh

    def _trace_ratio(self, eta):
        tr_fg = (self.components.T @ self.R_fg @ self.components).trace()

        # this is the way to add eta in cNRL by Fujiwara et al., 2020.
        # https://arxiv.org/abs/2005.12419
        # tr_bg = (self.components.T @ self.B_bg @ self.components +
        #          np.identity(self.components.shape[1]) * eta).trace()

        # here is the new way to add eta to make sure eta is the ratio of tr_fg
        tr_bg = (self.components.T @ self.R_bg
                 @ self.components).trace() + tr_fg * eta

        return np.asscalar(tr_fg) / np.asscalar(tr_bg)

    def fit(self,
            fg,
            bg,
            auto_alpha_selection=True,
            alpha=None,
            eta=1e-3,
            convergence_ratio=1e-2,
            max_iter=10,
            onehot_encoded=False,
            precision=np.float32,
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
        onehot_encoded: boolean, optional, (default=False)
            If True, fit() directly uses fg and bg as one-hot encoded matrices.
            Otherwise, fit() applies one-hot encoding to fg and bg before
            computing Burt matrices.
        precision: numpy dtype, optional, (default=np.float32)
            Computation precision. For example, if fast computation is needed
            rather than high precision, np.float16 can be set.
        y: dummy paramter
        Returns
        -------
        self.
        """
        G_fg = None
        G_bg = None
        if onehot_encoded:
            G_fg = fg
            G_bg = bg

            # generate input categories used for one-hot encoder from colnames
            self.categories_ = []
            prefix = None
            for col_name in G_fg.columns:
                if prefix != col_name.split('_')[-2]:
                    self.categories_.append([])
                    prefix = col_name.split('_')[-2]
                postfix = col_name.split('_')[-1]
                self.categories_[-1].append(postfix)
            for i, cate in enumerate(self.categories_):
                self.categories_[i] = np.array(cate, dtype='object')
        else:
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
                    np.concatenate(
                        (fg[col_name].unique(), bg[col_name].unique())))
                for col_name in fg.columns
            ]

            # apply one-hot encoder
            encoder = CMCA.OneHotEncoder(self.cate_each_q_).fit(fg)
            G_fg = encoder.transform(fg)
            G_bg = encoder.transform(bg)

        # Apply CA to the disjunctive matrices
        if alpha is None:
            alpha = 0
        super().fit(G_fg, G_bg, alpha=alpha, precision=precision, y=y)

        if auto_alpha_selection:
            new_alpha = self._trace_ratio(eta)
            while max_iter > 0 and new_alpha > alpha and (
                    new_alpha - alpha) / (alpha + 1e-15) > convergence_ratio:
                alpha = new_alpha
                self.update_fit(alpha)
                new_alpha = self._trace_ratio(eta)
                max_iter -= 1

        return self

    def update_fit(self, alpha):
        '''Update fit with a new alpha value. Unlike fit, this does not compute
        Burt matrices of fg and bg but utilize the Burt matrices already
        produced by applying fit once.

        Parameters
        ----------
        alpha: float
            A contrast parameter, which quantifies the trade-off between having
            high target variance and low background variance. alpha must be
            equal to or larger than 0. If 0, the result will be the same with
            the ordinary PCA. If auto_alpha_selection is True, this alpha is
            used as an initial alpha value for auto selection.
        Returns
        -------
        self.
        '''
        return super().update_fit(alpha)

    def _row_coordinates(self, X, onehot_encoded=False):
        G = None
        if onehot_encoded:
            G = X
        else:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            G = CMCA.OneHotEncoder(self.cate_each_q_).fit(X).transform(X)

        return super().row_coordinates(G)

    def _col_coordinates(self, X, onehot_encoded=False):
        G = None
        if onehot_encoded:
            G = X
        else:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            G = CMCA.OneHotEncoder(self.cate_each_q_).fit(X).transform(X)

        return super().col_coordinates(G)

    def transform(self, X, axis='row', onehot_encoded=False):
        """Compute row coordinates (or cloud of individuals) or column
        coordinates (or cloud of categories) with learned components by fit.

        Parameters
        ----------
        X: pandas dataframe, shape (n_samples, n_questions)
            A categorical dataset.
        axis: 'row' or 'col', optional, default='row'
            If 'row', compute row coordniates. Otherwise, compute column
            coordinates.
        onehot_encoded: boolean, optional, (default=False)
            If True, transform() directly uses X as one-hot encoded matrix.
            Otherwise, transform() applies one-hot encoding to X before
            obtaining row or column coordinates.
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
            X, onehot_encoded=onehot_encoded
        ) if axis == 'row' else self._col_coordinates(
            X, onehot_encoded=onehot_encoded)
