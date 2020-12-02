# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from os import path

import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
from googletrans import Translator
import copy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings


# %matplotlib inline
matplotlib.rcParams.update({"font.size": 22})

warnings.filterwarnings("ignore")

np.random.seed(seed=568329)

data_path = "/home/kendra/Downloads/Doing Open Science in Grad School (Responses) - Form Responses 1.csv"
survey = pd.read_csv(data_path, dtype=str)
# -

column_file_path = (
    "/home/kendra/Downloads/survey_column_names_and_translations.csv"
)
cols = pd.read_csv(column_file_path, index_col="variable_name", dtype=str)

# +
df = pd.DataFrame(columns=list(cols.index))
for col in list(cols.index):
    en, fr = cols.loc[col]
    df[col] = survey[en].fillna(survey[fr])


df["language"] = df["language"].fillna("English")
# df = df.fillna('[blank]')
df_cleaned = copy.deepcopy(df)
df.head(3)
# -

print("We have %d responses" % len(df))

# +
matplotlib.rcParams.update({"font.size": 18})


def plot_categorical_column(
    column, regex_dict, ax, title=" ", drop_other=False, kind="barh"
):
    keys = regex_dict.keys()
    column = column.dropna()
    for item, regex_str in regex_dict.items():
        column[column.str.contains(regex_str, regex=True)] = item

    if drop_other:
        for i, row in column.iteritems():
            if row not in keys:
                column = column.drop(i)

    column.value_counts(normalize=False).sort_values().plot(kind="barh")
    plt.title(title)
    return


# -

# ## Multiple choice questions

matplotlib.rcParams.update({"font.size": 18})
df["language"].value_counts().plot(
    kind="barh", title="Language used in the survey"
)

# ## University

# +

uni_regexs = {
    "McGill": "[Mm]c[Gg]ill",
    "Concordia": "[Cc]oncordia",
    "UdeM": "[Mm]ontr[eé]al|[Uu]de[Mm]",
}

fig, ax = plt.subplots(figsize=(7, 5))
plot_categorical_column(
    df["university"],
    uni_regexs,
    ax=ax,
    title="University",
    drop_other=False,
    kind="barh",
)
# -

# ## Current level of training

level_regexs = {
    "Undergrad": "[Uu]ndergrad|bac",
    "Masters": "[Mm]aster|[Mm]aitrise",
    "PhD": "[Pp]h[Dd]|[Dd]octora",
    "Post-Doc": "([Pp]ost)([Dd]oc)",
    "Professor": "[Pp]rof",
}
fig, ax = plt.subplots(figsize=(5, 5))
plot_categorical_column(
    df["level"],
    level_regexs,
    ax=ax,
    title="Current level of training",
    drop_other=True,
    kind="pie",
)

# ## Department

dept_regexs = {
    "Neuro": "[Nn]euro|[Ii][Pp][Nn]",
    "Psych": "[Pp]sych",
    "Bio": "[Bb]io",
    "Engineering": "[Ee]ngineering",
    "Medical": "([Mm]edical|[Hh]ealth|[Ii]nfirm)",
    "Cognitive": "[Cc]ognit",
    "Education": "[Ed]ucat",
    "[no answer]": "blank",
}
fig, ax = plt.subplots(figsize=(5, 5))
plot_categorical_column(
    df["department"],
    dept_regexs,
    ax=ax,
    title="Department",
    drop_other=False,
    kind="pie",
)

# ## What's your experience CREATING open-science objects?

# +
create_use_cols = [
    col for col in list(df.columns) if "create_" in col or "use_" in col
]
create_use_regexs = {
    "[blank]": "Non applicable",
    "dont_know": "I don't know what this is|Je ne sais pas ce que c'est.",
    "never_used": "I've never used this|Je n'ai jamais utilisé cela.",
    "have_used": "I've used this|J'ai utilisé ceci.",
    "like_to_use": "I'd like to use this|J'aimerais pouvoir utiliser cela.",
    "never_created": "I've never created this|Je n'ai jamais créé cela.",
    "have_created": "I've created this|J'ai créé ceci.",
    "like_to_create": "I'd like to create this|J'aimerais pouvoir créer cela.",
}

for key, regex_str in create_use_regexs.items():
    df_cleaned.replace(
        to_replace=regex_str,
        value=key,
        inplace=True,
        limit=None,
        regex=True,
        method="pad",
    )

# +
create_use_keys = list(create_use_regexs.keys())
mat = pd.DataFrame(columns=create_use_cols, index=create_use_keys, dtype=int)

for col in create_use_cols:
    for row in create_use_keys:
        mat.at[row, col] = df_cleaned[col].str.contains(row).sum()

mat[create_use_cols] = mat[create_use_cols].astype(int)


# +
def ticklabels(var_list):
    return [
        s.replace("use_", "")
        .replace("create_", "")
        .replace("_", " ")
        .replace("each cit", "each / cit")
        for s in var_list
    ]


matplotlib.rcParams.update({"font.size": 13})
fig, axs = plt.subplots(1, 2, figsize=(20, 5))

create_cols = [col for col in list(mat.columns) if "create_" in col]
create_keys = ["dont_know", "never_created", "have_created", "like_to_create"]
sns.heatmap(
    mat.loc[create_keys, create_cols].T, annot=True, vmax=len(df), ax=axs[0]
)
axs[0].set_title("Experience CREATING open-science objects", fontsize=20)
axs[0].set_yticklabels(ticklabels(create_cols))
axs[0].set_xticklabels(ticklabels(create_keys))

use_cols = [col for col in list(mat.columns) if "use_" in col]
use_keys = ["dont_know", "never_used", "have_used", "like_to_use"]
sns.heatmap(mat.loc[use_keys, use_cols].T, annot=True, vmax=len(df), ax=axs[1])
axs[1].set_title("Experience USING open-science objects", fontsize=20)
axs[1].set_yticklabels(ticklabels(use_cols))
axs[1].set_xticklabels(ticklabels(use_keys))

plt.tight_layout()
# -

# ## What barriers do you face when trying to do open science?

# +
barriers_regexs = {
    "Limited knowledge": "knowledge|connaissances",
    "Limited skills": "skills|Compétences",
    "Limited time": "time|temps",
    "Limited support": "support|Soutien",
    "Limited motivation": "[Mm]otivation",
    "Not convinced": "convinced|[Cc]onvaincu",
}

fig, ax = plt.subplots(figsize=(5, 5))
plot_categorical_column(
    df["barriers"],
    barriers_regexs,
    ax=ax,
    title="What barriers do you face when trying to do open science?",
    drop_other=True,
    kind="pie",
)


# -

# ## Would you be interested in applying to host Open Science Office Hours?

# +
#######################################################################
# -

# ## Open-ended questions

# +
def all_answers_wordcloud(column, title="", language="all"):
    if language == "en":
        column = column[df["language"] == "English"]
    elif language == "fr":
        column = column[df["language"] == "Français"]

    column = column.dropna()

    # Concatenate the texts from all participants
    all_texts = ""
    for text in column:
        all_texts = all_texts + " " + str(text)

    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(all_texts)

    # Display the generated image:
    wordcloud = WordCloud(
        max_font_size=50, max_words=100, background_color="white"
    ).generate(all_texts)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title + "\n", fontsize=30)
    plt.show()


# select language ('all', 'en', or 'fr')
language = "en"
all_answers_wordcloud(
    df["first_thoughts"],
    title='What do you think of when you hear the words "open science"?',
    language=language,
)

all_answers_wordcloud(
    df["learn_next"],
    title="What would you like to learn about open science?",
    language=language,
)

all_answers_wordcloud(
    df["motivation"],
    title="If you do open science,\nwhat motivates you to do so?",
    language=language,
)

all_answers_wordcloud(
    df["resources_that_helped"],
    title="If you do open science,\nwhat kinds of resources did you find most useful\nfor learning how to do open science?",
    language=language,
)
# -

