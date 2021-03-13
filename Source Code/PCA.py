import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from pca import pca
def process_data_pca(data):
    num_data = data.drop(['mfr', 'name', 'type', 'weight', 'shelf', 'cups'], 1)
    num_data.head()
    filt_d = num_data[num_data >= 0].dropna()
    X = filt_d
    y = filt_d['rating']
    return X,y

def fit_pca(X):
    sc = StandardScaler()
    pca = PCA()
    X_s = sc.fit_transform(X)
    X_t = pca.fit_transform(X_s)
    pca_val = {}
    for i in range(len(X.columns)):
        pca_val[X.columns[i]] = pca.components_[0][i]
    importance_feat = dict(sorted(pca_val.items(), key=lambda item: abs(item[1]), reverse=True))
    print("feature importance")
    print(importance_feat)
    top_feat = list(importance_feat.keys())[:6]
    return X_s,X_t, pca, top_feat, importance_feat

def pie_variance(pca):
    explained_variance = pca.explained_variance_ratio_
    tmp_var = np.append(explained_variance[:4], sum(explained_variance[4:]))
    pca_comp = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5-10']
    data = tmp_var

    # Creating explode data
    explode = (0.1, 0.0, 0.2, 0.1, 0.0)
    print(len(explode))
    # Creating color parameters
    colors = ("tomato", "coral", "brown",
              "grey", "lavender")
    print(len(colors))
    # Wedge properties
    wp = {'linewidth': 1}

    # Creating autocpt arguments
    def func(pct, allvalues):
        absolute = int(pct / 1. * np.sum(allvalues))
        return "{:.1f}%".format(pct, absolute)
    # Creating plot
    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: func(pct, data),
                                      explode=explode,
                                      labels=pca_comp,
                                      shadow=True,
                                      colors=colors,
                                      startangle=90,
                                      wedgeprops=wp,
                                      textprops=dict(color="black"))

    plt.setp(autotexts, size=7)
    ax.set_title("Explained Variance % per Dimension")
    # show plot
    plt.show()

def gen_bi_plot(score, coeff, top_feat, importance_feat, X,y):

        pop_a = mpatches.Patch(color='r', label='Strong Contribution')
        pop_b = mpatches.Patch(color='royalblue', label='Weak Contribution')

        plt.legend(handles=[pop_a, pop_b])
        plt.style.use(['default'])
        xs = score[:, 0]
        ys = score[:, 1]
        n = coeff.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        cmap = plt.get_cmap('jet', 12)
        # plt.scatter(xs * scalex,ys * scaley, c = y_train)
        plt.scatter(xs * scalex, ys * scaley, c=y)
        for i in range(n):
            # print(coeff[i,0])
            if X.columns[i] in top_feat:
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5, head_width=0.02)
            else:
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='royalblue', alpha=0.5, head_width=0.01)
            # plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = cmap[i], cmap="summer",alpha = 0.5,head_width = 0.02)
            mag = importance_feat[X.columns[i]]
            tmp = "\n ({:.2f})".format(mag)
            string = X.columns[i] + tmp
            # print(string)
            plt.text(coeff[i, 0] * 1.10, coeff[i, 1] * 1.10, string, color='g', ha='left', va='bottom')

def bi_plot(X_t, pca,top_feat, importance_feat, X,y):
    plt.figure(figsize=(12, 12))
    plt.xlim(-0.45, 0.75)
    plt.ylim(-0.68, 0.6)
    # plt.xlim(-5,5)
    # plt.ylim(-0.68,0.68)
    plt.xlabel("PC 1 ({})%".format(33.47))
    plt.ylabel("PC 2 ({})%".format(22.97))
    # plt.grid()

    # Call the function. Use only the 2 PCs.
    # myplot(X_train[:,0:2],np.transpose(pca.components_[0:2, :3]))
    gen_bi_plot(X_t[:, 0:2], np.transpose(pca.components_[0:2, :]), top_feat, importance_feat, X,y)
    plt.title("PCA Biplot")
    plt.show()

def three_dimension_plot(X_s, X):
    X_pca = pd.DataFrame(data=X_s, columns=list(X.columns))
    X_pca.reset_index(drop=True, inplace=True)
    #fit pca
    model = pca(n_components=0.95)
    model.fit_transform(X_pca)
    ax = model.biplot3d(n_feat=8, legend=False)
    return ax

if __name__ == '__main__':
    df = pd.read_csv("../Dataset/cereal.csv")
    # cleaning data
    X, y = process_data_pca(df)
    # fit pca
    X_s, X_t, model, top_feat, importance_feat = fit_pca(X)
    print("top features are")
    print(top_feat)
    pie_variance(model)
    bi_plot(X_t, model, top_feat, importance_feat, X, y)
    three_dimension_plot(X_s, X)