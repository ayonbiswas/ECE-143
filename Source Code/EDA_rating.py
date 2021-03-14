import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""Nitesh

Cereal Manufacturer Analysis:

1. Who are the major cereal manufacturers and what are their offerings?
2. What are their ratings?
3. What are their macro and micro nutrient stats?
4. Which manufacturer covers a wide variety of products and which manufactuere pertains to a niche market?

Who are the major cereal manufacturers and what are their offerings?
"""
def overview():
    #url = 'https://raw.githubusercontent.com/ayonbiswas/ECE-143/main/Dataset/cereal.csv'
    data = pd.read_csv("../Dataset/cereal.csv")

    #     data = pd.read_csv("../input/80-cereals/cereal.csv")
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

def plot_rating(mfr_group):

    plt.figure(figsize=(4, 3))
    mfr_group_rating = pd.DataFrame(mfr_group['rating'].count())
    mfr_group_rating.reset_index(inplace=True)
    mfr_group_rating = mfr_group_rating.sort_values("rating", ascending=False)
    sns.set_palette(sns.color_palette("muted"))
    sns.barplot(x=mfr_group_rating["rating"], y=mfr_group_rating["mfr"])
    plt.xlabel("# of products", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.title("Manufacturers", fontsize=15)
    plt.show()

    for mfr_name, mfr_products in mfr_group:
        mfr_products = mfr_products.sort_values("rating", ascending=False)
        plt.figure(figsize=(5, 4))
        sns.barplot(x=mfr_products["rating"], y=mfr_products["name"])
        plt.ylabel("Products", fontsize=12)
        plt.xlabel("Rating", fontsize=12)
        plt.title(mfr_name, fontsize=12)
        plt.show()

    plt.figure(figsize=(4, 3))
    mfr_group_rating = pd.DataFrame(mfr_group['rating'].mean())
    mfr_group_rating.reset_index(inplace=True)
    mfr_group_rating = mfr_group_rating.sort_values("rating", ascending=False)
    sns.set_palette(sns.color_palette("muted"))
    sns.barplot(x=mfr_group_rating["rating"], y=mfr_group_rating["mfr"])
    plt.xlabel("Avg. Rating", fontsize=12)
    plt.ylabel("", fontsize=12)
    plt.title("Manufacturers", fontsize=15)
    plt.show()

    for mfr_name, mfr_products in mfr_group:
        plt.figure(figsize=(4, 3))
        sns.distplot(mfr_products['rating'], kde=False, label=mfr_name)
        plt.title(mfr_name, fontsize=15)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Number of products', fontsize=12)
        plt.show()

def nutrient_plot(mfr_group):
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
    mfr_group_mean["calories (Kcal)"] = mfr_group_mean["calories"].map(lambda x: x)
    mfr_group_mean = mfr_group_mean.transpose()
    mfr_group_mean["AHFP"] = mfr_group_mean["American Home Food Products"]
    mfr_group_mean = mfr_group_mean.drop(columns=["American Home Food Products"])
    mfr_group_mean_macro = mfr_group_mean.drop(
        ["rating", "cups", "weight", "shelf", "vitamins", "potass", "sodium", "calories", "calories (Kcal)", "sugars",
         "fat", "carbo", "protein", "fiber", "vitamins (IU)", "potassium (mg)", "sodium (mg)"])

    mfr_group_mean_macro[list(mfr_group_mean_macro.keys())].T.plot(kind='bar', stacked=True, figsize=(4, 4),
                                                                   fontsize=12, legend=False)
    plt.title("Avg. Macro Nutrients Per Serving", fontsize=15)
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()

    mfr_group_mean_micro = mfr_group_mean.drop(
        ["rating", "cups", "weight", "shelf", "vitamins", "calories", "calories (Kcal)", "sugars",
         "fat", "carbo", "protein", "fiber", "sugars (g)", "fat (g)", "carbohydrates (g)", "protein (g)",
         "fiber (g)", "potass", "sodium"])
    mfr_group_mean_micro[list(mfr_group_mean_micro.keys())].T.plot(kind='bar', stacked=True, figsize=(4, 4),
                                                                   fontsize=12, legend=False)
    plt.title("Avg. Micro Nutrients Per Serving", fontsize=15)
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()

    mfr_group_mean_calories = mfr_group_mean.drop(
        ["rating", "cups", "weight", "shelf", "vitamins", "calories", "sugars",
         "fat", "carbo", "protein", "fiber", "sugars (g)", "fat (g)", "carbohydrates (g)", "protein (g)",
         "fiber (g)", "potass", "sodium", "potassium (mg)", "sodium (mg)", "vitamins (IU)"])
    mfr_group_mean_calories[list(mfr_group_mean_calories.keys())].T.plot(kind='bar', stacked=True, figsize=(3, 3),
                                                                         fontsize=12, legend=False)
    plt.title("Avg. Calories Per Serving", fontsize=15)
    plt.xlabel("")
    plt.ylabel("")
    plt.show()


def plot_niche(mfrgroup):
    """Which manufacturer covers a wide variety of products and which manufacturer pertains to a niche market?"""

    for mfr_name, mfr_products in mfr_group:
        req_vars = ["carbo", "protein", "fat"]
        req_var_names = ["Carbohydrates (g)", "Proteins (g)", "Fat (g)"]
        plt.figure(figsize=(4, 4))
        for i, req_var in enumerate(req_vars):
            sns.distplot(mfr_products[req_var], kde=False, label=req_var_names[i])
        plt.legend()
        plt.title(mfr_name, fontsize=15)
        plt.xlabel('Nutrients (g)', fontsize=12)
        plt.ylabel('Number of products', fontsize=12)
        plt.show()

if __name__ == '__main__':
    data = overview()
    mfr_map = {"A": "American Home Food Products", "G": "General Mills", "K": "Kelloggs", "N": "Nabisco", "P": "Post",
               "Q": "Quaker Oats", "R": "Ralston Purina"}
    data_2 = data.copy()
    data_2["mfr"] = data_2["mfr"].map(lambda x: mfr_map[x])
    mfr_group = data_2.groupby('mfr')
    plot_rating(mfr_group)
    plot_niche(mfr_group)