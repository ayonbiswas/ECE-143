# -*- coding: utf-8 -*-
"""data-visualisation
Original file is located at
    https://colab.research.google.com/drive/1cS6QeivJbTwO4XWm0ojJC7QOOKzTVp8I
"""

import numpy as np                
import pandas as pd               
import seaborn as sns             
import matplotlib.pyplot as plt   
import scipy.stats               
from sklearn import preprocessing
import plotly.express as px
import os

url = 'https://raw.githubusercontent.com/ayonbiswas/ECE-143/main/Dataset/cereal.csv'
df = pd.read_csv(url)

df.head()

from google.colab import files

f, axes = plt.subplots(1,2, figsize=(10, 5))
sns.countplot(x="mfr", data=df, ax=axes[0], palette="Set3")
sns.countplot(x="type", data=df, ax=axes[1], palette="Set2")
plt.savefig("/content/drivecount.png", dpi = 150)

corr=df.iloc[:,~df.columns.isin(['name','mfr','type'])].corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=2, cbar_kws={"shrink": .5})


plt.savefig("/content/corr.png", dpi = 150)

"""Calories have notable positive correlations with weight, sugars and fat.
Strong negative correlations between sugars and rating, and calories and rating.
"""

# f, axes = plt.subplots(1,2, figsize=(10, 5))
sns.lmplot(x="sugars",y="rating",data=df)
plt.title('Sugars vs Rating', size= 16)
plt.xlabel("grams of sugars", size = 14)
plt.ylabel("rating", size =14)
plt.savefig("/content/sugar_rating.png", dpi = 150, bbox_inches='tight')

sns.lmplot(x="calories", y="rating", data=df)
plt.title('Calories vs Rating', size=16)
plt.ylabel("rating", size = 14)
plt.xlabel("calories per serving", size =14)
plt.savefig("/content/calories_rating.png", dpi = 150, bbox_inches='tight')
# sns.jointplot(x="calories", y="rating", data=df)

"""cereals with high calories or sugar are less preferred."""

sns.lmplot(x="protein", y="rating", data=df)
plt.title('Protein vs Rating', size= 16)
plt.xlabel("grams of protein", size = 14)
plt.ylabel("rating", size =14)
plt.savefig("/content/protein_rating.png", dpi = 150, bbox_inches='tight')

sns.lmplot(x="fiber", y="rating", data=df)
plt.title('Fiber vs Rating', size= 16)
plt.xlabel("grams of dietary fiber", size = 14)
plt.ylabel("rating", size =14)
plt.savefig("/content/fiber_rating.png", dpi = 150, bbox_inches='tight')
# sns.lmplot(x="vitamins", y="rating", data=df)

"""health is preferred over taste by customers in general

# comparing good vs bad
"""

cereals_scale = df

scaler = preprocessing.StandardScaler()
columns =df.columns[3:]
cereals_scale[columns] = scaler.fit_transform(cereals_scale[columns])
cereals_scale.head()

cereals_scale['Good'] = cereals_scale.loc[:,['protein','fiber','vitamins']].mean(axis=1)

cereals_scale['Bad'] = cereals_scale.loc[:,['fat','sodium','potass', 'sugars']].mean(axis=1)

ax = sns.lmplot('Good', 
           'Bad',
           data=cereals_scale, 
           fit_reg=True,
           height = 10,
           aspect =2 )

# plt.title('Cereals Plot')
plt.xlabel('Good', size = 16)
plt.ylabel('Bad', size =16)


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
          ax.text(point['x']+.02, point['y'], str(point['val']), size = 12)

label_point(cereals_scale.Good, cereals_scale.Bad, cereals_scale.name, plt.gca()) 

plt.savefig("/content/good_bad.png", dpi = 150, bbox_inches='tight')



"""Zongcheng Wang

# Can type of cereal (hot or cold) affect rating?

my reference https://www.kaggle.com/kianwee/analysis-on-cereal-prediction-on-ratings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def overview():
    
    url = 'https://raw.githubusercontent.com/ayonbiswas/ECE-143/main/Dataset/cereal.csv'
    data = pd.read_csv(url)
    
    print("The first 5 rows of data are:\n")
    print(data.head)
    print("\n\n\nDataset has {} rows and {} columns".format(data.shape[0], data.shape[1]))
    print("\n\n\nDatatype: \n")
    print(data.dtypes)
    print("\n\n\nThe number of null values for each column are: \n")
    print(data.isnull().sum())
    print("\n\n\nData summary: \n")
    print(data.describe())
    return data
    
data = overview()

# Count the number of -1 in carbo, sugars and potass column
data[data == -1].count(axis=0)

# Count the number of -1 in carbo, sugars and potass column
data[data == -1].count(axis=0)

# Remove affected rows
data = data[(data.carbo >= 0) & (data.sugars >= 0) & (data.potass >= 0)]
data[data == -1].count(axis=0)

# Counting number of manufacturers 
data['mfr'].value_counts()

plt.figure(figsize = (10, 8))
sns.boxplot(data = data, x = "mfr", y = "rating")

# Best rating cereal
data.loc[data['rating'] == max(data.rating)]





"""# Can macronutrients and calories affect ratings?"""

fig = plt.figure()

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams["font.weight"] = "bold"

fontdict={'fontsize': 25,
          'weight' : 'bold'}

fontdicty={'fontsize': 18,
          'weight' : 'bold',
          'verticalalignment': 'baseline',
          'horizontalalignment': 'center'}

fontdictx={'fontsize': 18,
          'weight' : 'bold',
          'horizontalalignment': 'center'}

plt.subplots_adjust(wspace=0.4, hspace=0.3)


ax1 = fig.add_subplot(221)
ax1.scatter('calories', 'rating', data= data, c="green")
ax1.set_title('Calories', fontdict=fontdict, color="green")
plt.ylabel('Rating')
plt.xlabel('grams per serving')


ax2 = fig.add_subplot(222)
ax2.scatter('fat', 'rating', data=data, c="orange")
ax2.set_title('Fat', fontdict=fontdict, color="orange")
plt.xlabel('grams per serving')

ax3 = fig.add_subplot(223)
ax3.scatter('protein', 'rating', data=data, c="brown")
ax3.set_title('Protein', fontdict=fontdict, color="brown")
plt.xlabel('grams per serving')
plt.ylabel('Rating')

ax4 = fig.add_subplot(224)
ax4.scatter('carbo', 'rating', data=data, c="blue")
ax4.set_title("Carbs", fontdict=fontdict, color="blue")
plt.xlabel('grams per serving')

"""# Can micronutrients affect ratings?

"""

fig = plt.figure()

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams["font.weight"] = "bold"

fontdict={'fontsize': 25,
          'weight' : 'bold'}

fontdicty={'fontsize': 18,
          'weight' : 'bold',
          'verticalalignment': 'baseline',
          'horizontalalignment': 'center'}

fontdictx={'fontsize': 18,
          'weight' : 'bold',
          'horizontalalignment': 'center'}

plt.subplots_adjust(wspace=0.4, hspace=0.3)

# fig.suptitle('Can micronutrients affect cereal rating?', fontsize=25,fontweight="bold", color="black", 
#              position=(0.5,1.01))

ax1 = fig.add_subplot(221)
ax1.scatter('sodium', 'rating', data= data, c="green")
ax1.set_title('Sodium', fontdict=fontdict, color="green")
plt.ylabel('Rating')
plt.xlabel('grams per serving')

ax2 = fig.add_subplot(222)
ax2.scatter('fat', 'rating', data=data, c="orange")
ax2.set_title('Potassium', fontdict=fontdict, color="orange")
plt.xlabel('grams per serving')

ax3 = fig.add_subplot(223)
ax3.scatter('fiber', 'rating', data=data, c="brown")
ax3.set_title('Fiber', fontdict=fontdict, color="brown")
plt.ylabel('Rating')
plt.xlabel('grams per serving')


ax4 = fig.add_subplot(224)
ax4.scatter('vitamins', 'rating', data=data, c="blue")
ax4.set_title("Vitamins", fontdict=fontdict, color="blue")
plt.xlabel('grams per serving')



"""Nitesh

Cereal Manufacturer Analysis:

1. Who are the major cereal manufacturers and what are their offerings?
2. What are their ratings?
3. What are their macro and micro nutrient stats?
4. Which manufacturer covers a wide variety of products and which manufactuere pertains to a niche market?

Who are the major cereal manufacturers and what are their offerings?
"""

mfr_map = {"A":"American Home Food Products","G":"General Mills","K":"Kelloggs","N":"Nabisco","P":"Post","Q":"Quaker Oats","R":"Ralston Purina"}
data_2 = data.copy()
data_2["mfr"] = data_2["mfr"].map(lambda x: mfr_map[x])
mfr_group = data_2.groupby('mfr')

plt.figure(figsize=(4,3))
mfr_group_rating = pd.DataFrame(mfr_group['rating'].count())
mfr_group_rating.reset_index(inplace=True)
mfr_group_rating = mfr_group_rating.sort_values("rating",ascending=False)
sns.set_palette(sns.color_palette("muted"))
sns.barplot(x=mfr_group_rating["rating"],y=mfr_group_rating["mfr"])
plt.xlabel("# of products",fontsize=12)
plt.ylabel("",fontsize=12)
plt.title("Manufacturers",fontsize=15)
plt.show()

for mfr_name, mfr_products in mfr_group:
  mfr_products = mfr_products.sort_values("rating", ascending=False)
  plt.figure(figsize=(5,4))
  sns.barplot(x=mfr_products["rating"],y=mfr_products["name"])
  plt.ylabel("Products",fontsize=12)
  plt.xlabel("Rating",fontsize=12)
  plt.title(mfr_name,fontsize=12)
  plt.show()

"""What are their ratings?"""

plt.figure(figsize=(4,3))
mfr_group_rating = pd.DataFrame(mfr_group['rating'].mean())
mfr_group_rating.reset_index(inplace=True)
mfr_group_rating = mfr_group_rating.sort_values("rating",ascending=False)
sns.set_palette(sns.color_palette("muted"))
sns.barplot(x=mfr_group_rating["rating"],y=mfr_group_rating["mfr"])
plt.xlabel("Avg. Rating",fontsize=12)
plt.ylabel("",fontsize=12)
plt.title("Manufacturers",fontsize=15)
plt.show()

for mfr_name, mfr_products in mfr_group:
  plt.figure(figsize=(4,3))
  sns.distplot(mfr_products['rating'],  kde=False, label=mfr_name)
  plt.title(mfr_name, fontsize=15)
  plt.xlabel('Rating',fontsize=12)
  plt.ylabel('Number of products',fontsize=12)
  plt.show()

data_2

"""What are their macro and micro nutrient stats?

"""

sns.set()
mfr_group_mean = pd.DataFrame(mfr_group.mean())
mfr_group_mean["sugars (g)"] = mfr_group_mean["sugars"]
mfr_group_mean["fat (g)"] = mfr_group_mean["fat"]
mfr_group_mean["carbohydrates (g)"] = mfr_group_mean["carbo"]
mfr_group_mean["protein (g)"] = mfr_group_mean["protein"]
mfr_group_mean["sodium (mg)"] = mfr_group_mean["sodium"]
mfr_group_mean["potassium (mg)"] = mfr_group_mean["potass"]
mfr_group_mean["vitamins (IU)"] = mfr_group_mean["vitamins"].map(lambda x: x)
mfr_group_mean["fiber (g)"] = mfr_group_mean["fiber"]
mfr_group_mean["calories (Kcal)"] = mfr_group_mean["calories"].map(lambda x:x)
mfr_group_mean = mfr_group_mean.transpose()
mfr_group_mean["AHFP"] = mfr_group_mean["American Home Food Products"]
mfr_group_mean = mfr_group_mean.drop(columns=["American Home Food Products"])
mfr_group_mean_macro = mfr_group_mean.drop(["rating","cups","weight","shelf","vitamins","potass","sodium","calories","calories (Kcal)","sugars",
                                            "fat","carbo","protein","fiber","vitamins (IU)","potassium (mg)", "sodium (mg)"])

mfr_group_mean_macro[list(mfr_group_mean_macro.keys())].T.plot(kind='bar', stacked=True, figsize=(4,4), fontsize=12, legend=False)
plt.title("Avg. Macro Nutrients Per Serving", fontsize=15)
plt.xlabel("")
plt.ylabel("")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

mfr_group_mean_micro = mfr_group_mean.drop(["rating","cups","weight","shelf","vitamins","calories","calories (Kcal)","sugars",
                                            "fat","carbo","protein","fiber","sugars (g)","fat (g)","carbohydrates (g)","protein (g)",
                                            "fiber (g)","potass","sodium"])
mfr_group_mean_micro[list(mfr_group_mean_micro.keys())].T.plot(kind='bar', stacked=True, figsize=(4,4), fontsize=12, legend=False)
plt.title("Avg. Micro Nutrients Per Serving", fontsize=15)
plt.xlabel("")
plt.ylabel("")
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

mfr_group_mean_calories = mfr_group_mean.drop(["rating","cups","weight","shelf","vitamins","calories","sugars",
                                            "fat","carbo","protein","fiber","sugars (g)","fat (g)","carbohydrates (g)","protein (g)",
                                            "fiber (g)","potass","sodium", "potassium (mg)", "sodium (mg)","vitamins (IU)" ])
mfr_group_mean_calories[list(mfr_group_mean_calories.keys())].T.plot(kind='bar', stacked=True, figsize=(3,3), fontsize=12, legend=False)
plt.title("Avg. Calories Per Serving", fontsize=15)
plt.xlabel("")
plt.ylabel("")
plt.show()

"""Which manufacturer covers a wide variety of products and which manufacturer pertains to a niche market?"""

for mfr_name, mfr_products in mfr_group:
  req_vars = ["carbo","protein","fat"]
  req_var_names = ["Carbohydrates (g)", "Proteins (g)", "Fat (g)"]
  plt.figure(figsize=(4,4))
  for i, req_var in enumerate(req_vars):
    sns.distplot(mfr_products[req_var],  kde=False, label=req_var_names[i])
  plt.legend()
  plt.title(mfr_name, fontsize=15)
  plt.xlabel('Nutrients (g)',fontsize=12)
  plt.ylabel('Number of products', fontsize=12)
  plt.show()

"""**What are the factors that made each cereal different from the others (A PCA Analysis)**"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pca

from PCA import process_data_pca,fit_pca,pie_variance,gen_bi_plot, bi_plot,three_dimension_plot

X,y = process_data_pca(df)

print(X)

X_s,X_t, model, top_feat, importance_feat = fit_pca(X)

print("top features are")
print(top_feat)

pie_variance(model)

bi_plot(X_t, model, top_feat, importance_feat, X,y)

three_dimension_plot(X_s, X)