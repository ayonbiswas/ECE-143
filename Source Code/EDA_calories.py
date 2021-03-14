import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

'''
Ayon

Calories have notable positive correlations with weight, sugars and fat.
Strong negative correlations between sugars and rating, and calories and rating.

'''


def gen_corr(df):
    """Calories have notable positive correlations with weight, sugars and fat.
    Strong negative correlations between sugars and rating, and calories and rating.
    """
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.countplot(x="mfr", data=df, ax=axes[0], palette="Set3")
    sns.countplot(x="type", data=df, ax=axes[1], palette="Set2")
    #plt.savefig("/content/drivecount.png", dpi=150)

    corr = df.iloc[:, ~df.columns.isin(['name', 'mfr', 'type'])].corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=2, cbar_kws={"shrink": .5})

    #plt.savefig("/content/corr.png", dpi=150)
    # f, axes = plt.subplots(1,2, figsize=(10, 5))
    sns.lmplot(x="sugars", y="rating", data=df)
    plt.title('Sugars vs Rating', size=16)
    plt.xlabel("grams of sugars", size=14)
    plt.ylabel("rating", size=14)
    #plt.savefig("/content/sugar_rating.png", dpi=150, bbox_inches='tight')

    sns.lmplot(x="calories", y="rating", data=df)
    plt.title('Calories vs Rating', size=16)
    plt.ylabel("rating", size=14)
    plt.xlabel("calories per serving", size=14)
    #plt.savefig("/content/calories_rating.png", dpi=150, bbox_inches='tight')
    # sns.jointplot(x="calories", y="rating", data=df)

    """cereals with high calories or sugar are less preferred."""

    sns.lmplot(x="protein", y="rating", data=df)
    plt.title('Protein vs Rating', size=16)
    plt.xlabel("grams of protein", size=14)
    plt.ylabel("rating", size=14)
    #plt.savefig("/content/protein_rating.png", dpi=150, bbox_inches='tight')

    sns.lmplot(x="fiber", y="rating", data=df)
    plt.title('Fiber vs Rating', size=16)
    plt.xlabel("grams of dietary fiber", size=14)
    plt.ylabel("rating", size=14)
   # plt.savefig("/content/fiber_rating.png", dpi=150, bbox_inches='tight')
    # sns.lmplot(x="vitamins", y="rating", data=df)

    plt.show()
"""health is preferred over taste by customers in general

# comparing good vs bad
"""


def health_taste_plot(df):
    cereals_scale = df

    scaler = preprocessing.StandardScaler()
    columns = df.columns[3:]
    cereals_scale[columns] = scaler.fit_transform(cereals_scale[columns])
    cereals_scale.head()

    cereals_scale['Good'] = cereals_scale.loc[:, ['protein', 'fiber', 'vitamins']].mean(axis=1)

    cereals_scale['Bad'] = cereals_scale.loc[:, ['fat', 'sodium', 'potass', 'sugars']].mean(axis=1)

    ax = sns.lmplot('Good',
                    'Bad',
                    data=cereals_scale,
                    fit_reg=True,
                    height=10,
                    aspect=2)

    # plt.title('Cereals Plot')
    plt.xlabel('Good', size=16)
    plt.ylabel('Bad', size=16)
    label_point(cereals_scale.Good, cereals_scale.Bad, cereals_scale.name, plt.gca())
    #plt.savefig("/content/good_bad.png", dpi=150, bbox_inches='tight')
    plt.show()


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + .02, point['y'], str(point['val']), size=12)


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/ayonbiswas/ECE-143/main/Dataset/cereal.csv'
    df = pd.read_csv(url)
    df.head()
    gen_corr(df)
    health_taste_plot(df)