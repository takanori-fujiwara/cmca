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
        self.R_fg = None
        self.R_tg = None
        self.R = None
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

    def _standardized_residuals(self, X, precision):
        # Compute the correspondence matrix which contains the relative frequencies
        X = X / np.sum(X)

        # Compute standardized Burt Matrices
        r = X.sum(axis=1)
        c = X.sum(axis=0)
        c = np.asarray(c).reshape(X.shape[1], )
        for i in range(len(c)):
            c[i] = max(np.finfo(precision).tiny, c[i])

        rc = np.outer(r, c)
        D = sparse.diags(c**-0.5)
        R = D @ (X - rc).T @ (X - rc) @ D

        return R

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

        # Compute standardized Burt Matrices
        self.R_fg = self._standardized_residuals(fg, precision)
        self.R_bg = self._standardized_residuals(bg, precision)

        # Burt matrix for constrastive analyss
        self.R = self.R_fg - alpha * self.R_bg

        # Perform EVD (in our case, we can use Schur decomp)
        # self.w_, self.v_ = linalg.eig(self.B)
        schur_form, self.v_ = linalg.schur(self.R)
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
        self.R = self.R_fg - alpha * self.R_bg

        # Perform EVD
        self.w_, self.v_ = linalg.eig(self.R)
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

    def _standardize_disjuctive_mat(self, X):
        X = X / np.sum(X)

        c_masses = np.array(X.sum(axis=0).tolist()[0])
        # because some of fg or bg does not have any 1 value for some category
        # we need to fulfill tiny values
        c_masses[c_masses <= np.finfo(float).tiny] = np.finfo(float).tiny

        # Normalize the rows so that they sum up to 1
        if isinstance(X, np.ndarray):
            X = X / X.sum(axis=1)[:, None]
        else:
            X = X / X.sum(axis=1)

        X = X @ sparse.diags(c_masses**-0.5)

        return X

    def row_coordinates(self, X):
        """The row principal coordinates."""
        utils.validation.check_is_fitted(self)

        row_names, _ = self._row_col_names(X)

        if isinstance(X, pd.DataFrame):
            try:
                X = X.sparse.to_coo().astype(float)
            except AttributeError:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        X = self._standardize_disjuctive_mat(X)

        return pd.DataFrame(data=X @ self.components, index=row_names)

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
        Y_col = sparse.diags(c_masses_a**-1) @ X.T @ Y_row @ sparse.diags(
            1 / np.sqrt(np.abs(self.w_[:self.n_components])))

        return pd.DataFrame(Y_col, index=col_names)
