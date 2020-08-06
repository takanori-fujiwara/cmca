import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cmca import CMCA

# Congressional Voting Records Data Set
# https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
df = pd.read_csv('./data/house-votes-84.data', header=None)
with open('./data/house-votes-84.col_names', 'r') as f:
    # chr(10) is newline (to avoid newline when generating doc with sphinx)
    df.columns = [line.replace(chr(10), '') for line in f]
X = df.iloc[:, 1:]
y = np.array(df.iloc[:, 0])

fg = X.iloc[y == 'democrat']
bg = X.iloc[y == 'republican']

# alpha = 0 (normal MCA on fg)
# alpha = 10 (contrastive MCA fg vs bg)
cmca = CMCA(n_components=2)
for alpha in [0, 10, 'auto']:
    ### cMCA
    auto_alpha = False
    if alpha == 'auto':
        alpha = None
        auto_alpha = True
    cmca.fit(fg=fg, bg=bg, alpha=alpha, auto_alpha_selection=auto_alpha)

    # row coordinates (cloud of individuals)
    Y_fg_row = np.array(cmca.transform(fg, axis='row'))
    Y_bg_row = np.array(cmca.transform(bg, axis='row'))

    # col coordinates (cloud of categories)
    Y_fg_col = np.array(cmca.transform(fg, axis='col'))
    Y_bg_col = np.array(cmca.transform(bg, axis='col'))

    # cPC loadings
    loadings = cmca.loadings

    # category names
    categories = cmca.categories

    ### Plot the results
    alpha = int(cmca.alpha * 100) / 100
    plt.figure(figsize=[8, 8])
    # plot row coordinates
    plt.subplot(2, 2, 1)
    plt.scatter(Y_fg_row[:, 0], Y_fg_row[:, 1], c='b', s=5, label='demo')
    plt.scatter(Y_bg_row[:, 0], Y_bg_row[:, 1], c='r', s=5, label='rep')
    plt.legend(loc='best', shadow=False, scatterpoints=1, fontsize=10)
    plt.title(f'cMCA row coords. alpha = {alpha}')
    # plot col coordinates
    plt.subplot(2, 2, 2)
    plt.scatter(Y_fg_col[:, 0], Y_fg_col[:, 1], c='g', s=5, label='cate')
    for i in range(Y_fg_col.shape[0]):
        plt.text(Y_fg_col[i, 0], Y_fg_col[i, 1], str(i), fontsize=6)
    plt.legend(loc='best', shadow=False, scatterpoints=1, fontsize=10)
    plt.title(f'cMCA col coords. alpha = {alpha}')
    # plot cPC loadings
    plt.subplot(2, 2, 3)
    plt.scatter(loadings[:, 0], loadings[:, 1], c='g', s=5, label='cate')
    for i in range(loadings.shape[0]):
        plt.text(loadings[i, 0], loadings[i, 1], str(i), fontsize=6)
    plt.legend(loc='best', shadow=False, scatterpoints=1, fontsize=10)
    plt.title(f'cPC loadings. alpha = {alpha}')
    # plot category names
    plt.subplot(2, 2, 4)
    for i, cate in enumerate(categories):
        plt.text(0.1, i, str(i) + ': ' + cate, fontsize=6, c='g')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, 1])
    plt.ylim([len(categories) + 1, -1])
    plt.title('Categories')

    plt.tight_layout()
    plt.show()
