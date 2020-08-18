import numpy as np
import pandas as pd
from scipy import sparse
from scipy import linalg
from sklearn import utils


class CCA():
    '''
    Contrastive Correspondence Analysis (cCA). We referred to CA
    in Prince library (https://github.com/MaxHalford/prince) for this
    implementation. We modified the part related to perform contrastive analysis.
    '''
    def __init__(self, n_components=2, copy=True, check_input=False):
        self.n_components = n_components
        self.copy = copy
        self.check_input = check_input
        self.categories = None
        self.components = None
        self.loadings = None
        self.alpha = None
        self.B_fg = None
        self.B_tg = None
        self.B = None
        self.w_ = None
        self.v_ = None

    def _row_col_names(self, X):
        if isinstance(X, pd.DataFrame):
            row_names = X.index.tolist()
            col_names = X.columns.tolist()
        else:
            row_names = list(range(X.shape[0]))
            col_names = list(range(X.shape[1]))

        return row_names, col_names

    def fit(self, fg, bg, alpha, precision=np.float32, y=None):
        # Check input
        if self.check_input:
            utils.check_array(fg, accept_sparse=True)
            utils.check_array(bg, accept_sparse=True)

            # Check all values are positive
            if (fg < 0).any().any():
                raise ValueError("All values in fg should be positive")
            if (bg < 0).any().any():
                raise ValueError("All values in bg should be positive")

        self.alpha = alpha
        fg_r_names, fg_c_names = self._row_col_names(fg)
        bg_r_names, bg_c_names = self._row_col_names(bg)

        if not fg_c_names == bg_c_names:
            raise ValueError('fg\'s and bg\'s col names must match')
        self.categories = fg_c_names

        if self.copy:
            fg = np.copy(fg)
            bg = np.copy(bg)
        else:
            if isinstance(fg, pd.DataFrame):
                fg = fg.to_numpy()
            if isinstance(bg, pd.DataFrame):
                bg = bg.to_numpy()
        fg = fg.astype(precision)
        bg = bg.astype(precision)

        # without changing to csr matrix, Burt matrix comp is bit faster
        # fg = sparse.csr_matrix(fg)
        # bg = sparse.csr_matrix(bg)

        # Compute the correspondence matrix which contains the relative frequencies
        fg = fg / (np.sum(fg) + np.finfo(precision).tiny)
        bg = bg / (np.sum(bg) + np.finfo(precision).tiny)

        # Compute standardized Burt Matrices
        fg_r_masses = fg.sum(axis=1)
        fg_c_masses = fg.sum(axis=0)
        bg_r_masses = bg.sum(axis=1)
        bg_c_masses = bg.sum(axis=0)
        fg_c_masses = np.asarray(fg_c_masses).reshape(fg.shape[1], )
        bg_c_masses = np.asarray(bg_c_masses).reshape(bg.shape[1], )

        for i in range(len(fg_c_masses)):
            fg_c_masses[i] = max(np.finfo(precision).tiny, fg_c_masses[i])
        for i in range(len(bg_c_masses)):
            bg_c_masses[i] = max(np.finfo(precision).tiny, bg_c_masses[i])

        fg -= np.outer(fg_r_masses, fg_c_masses)
        bg -= np.outer(bg_r_masses, bg_c_masses)

        self.B_fg = sparse.diags(
            fg_c_masses**-0.5) @ fg.transpose() @ fg @ sparse.diags(fg_c_masses
                                                                    **-0.5)
        self.B_bg = sparse.diags(
            bg_c_masses**-0.5) @ bg.transpose() @ bg @ sparse.diags(bg_c_masses
                                                                    **-0.5)

        # Burt matrix for constrastive analyss
        self.B = self.B_fg - alpha * self.B_bg

        # Perform EVD (in our case, we can use Schur decomp)
        # self.w_, self.v_ = linalg.eig(self.B)
        schur_form, self.v_ = linalg.schur(self.B)
        self.w_ = linalg.eigvals(schur_form)

        top_eigen_indices = np.argsort(-self.w_)
        self.w_ = self.w_[top_eigen_indices]

        self.components = self.v_[:, top_eigen_indices[:self.n_components]]
        self.loadings = self.components * np.sqrt(np.abs(self.w_))[:,
                                                                   np.newaxis]

        return self

    def update_fit(self, alpha):
        '''
        Fit by updating only the part related to alpha value.
        '''
        self.alpha = alpha

        # Burt matrix for constrastive analyss
        self.B = self.B_fg - alpha * self.B_bg

        # Perform EVD
        self.w_, self.v_ = linalg.eig(self.B)
        top_eigen_indices = np.argsort(-self.w_)
        self.w_ = self.w_[top_eigen_indices]

        self.components = self.v_[:, top_eigen_indices[:self.n_components]]
        self.loadings = self.components * np.sqrt(np.abs(self.w_))[:,
                                                                   np.newaxis]

        return self

    def transform(self, X):
        """Computes the row principal coordinates of a dataset.

        Same as calling `row_coordinates`. In most cases you should be using the same
        dataset as you did when calling the `fit` method. You might however also want to included
        supplementary data.

        """
        utils.validation.check_is_fitted(self)
        if self.check_input:
            utils.check_array(X)
        return self.row_coordinates(X)

    def row_coordinates(self, X):
        """The row principal coordinates."""
        utils.validation.check_is_fitted(self)

        row_names, col_names = self._row_col_names(X)

        if isinstance(X, pd.DataFrame):
            try:
                X = X.sparse.to_coo().astype(float)
            except AttributeError:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        # Compute column masses
        X = X / np.sum(X)
        c_masses = pd.Series(np.squeeze(np.asarray(X.sum(axis=0))),
                             index=col_names).to_numpy(dtype=float)
        # because some of fg or bg does not have any 1 value for some category
        # we need to fulfill tiny values
        c_masses[c_masses <= np.finfo(float).tiny] = np.finfo(float).tiny

        # Normalize the rows so that they sum up to 1
        if isinstance(X, np.ndarray):
            X = X / X.sum(axis=1)[:, None]
        else:
            X = X / X.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(c_masses**-0.5) @ self.components,
            index=row_names)

    def col_coordinates(self, X):
        """The col principal coordinates. Compute row_coordinates first. Then, applying transiton formula"""
        utils.validation.check_is_fitted(self)

        row_names, col_names = self._row_col_names(X)

        if isinstance(X, pd.DataFrame):
            try:
                X = X.sparse.to_coo().astype(float)
            except AttributeError:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        # Compute column masses
        c_masses_a = pd.Series(np.squeeze(np.asarray(X.sum(axis=0))),
                               index=col_names).to_numpy(dtype=float)
        c_masses_a[c_masses_a <= np.finfo(float).tiny] = np.finfo(float).tiny
        X = X / np.sum(X)
        c_masses_b = pd.Series(np.squeeze(np.asarray(X.sum(axis=0))),
                               index=col_names).to_numpy(dtype=float)
        c_masses_b[c_masses_b <= np.finfo(float).tiny] = np.finfo(float).tiny

        # Normalize the rows so that they sum up to 1
        if isinstance(X, np.ndarray):
            X = X / X.sum(axis=1)[:, None]
        else:
            X = X / X.sum(axis=1)

        Y_row = X @ sparse.diags(c_masses_b**-0.5) @ self.components
        Y_col = sparse.diags(
            c_masses_a**-1) @ X.transpose() @ Y_row @ sparse.diags(
                1 / np.sqrt(np.abs(self.w_[:self.n_components])))

        return pd.DataFrame(Y_col, index=col_names)
