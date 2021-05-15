# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Analyzing State Vaccination Rate and 2020 Election Results
#
# Daniel Rowe
# %% [markdown]
# ## Table of Contents
# - [1. Introduction](#1-introduction)
# - [2. Set Up](#2-set-up)
# - [3. Data Wrangling](#3-data-wrangling)
#   - [3.1. Introduction to Datasets](#31-introduction-to-datasets)
#     - [3.1.1. Election Data](#311-election-data)
#     - [3.1.2. Vaccination Data](#312-vaccination-data)
#   - [3.2. Tidying the Data](#32-tidying-the-data)
# - [4. Exploratory Data Analysis](#4-exploratory-data-analysis)
# - [5. Hypothesis Testing and Machine Learning](#5-hypothesis-testing-and-machine-learning)
# - [6. Conclusion](#6-conclusion)
# %% [markdown]
# ## 1. Introduction
#
# Since the start of the COVID pandemic, there has been much debate on aspects of the virus. There has been speculation on the severity of the virus, whether or not one political party was [using the election to gain an edge](https://www.forbes.com/sites/jackbrewster/2020/08/24/trump-claims-democrats-are-using-covid-to-steal-the-election-in-first-convention-speech/). COVID caused the election to be handled differently with an increase in mail-in voters and [affected the confidence of many voters](https://www.forbes.com/sites/jackbrewster/2020/08/24/trump-claims-democrats-are-using-covid-to-steal-the-election-in-first-convention-speech/). Additionally, the divided opinions on social media and platform censorship sparked more division amongst individuals.
#
# After the election, social media opinions still differed and as the vaccine rolled out more speculation and conspiracies arose. Some people refused to wear masks while others argued for tighter restrictions. States handled restrictions and distributions of the vaccine differently but what does the data say about the election results and the vaccination rates?
# %% [markdown]
# ## 2. Set Up
#
# - Pandas: To display and organize the data
# - Numpy: Used for calculations
# - Matplotlib: Data visualization
# - Scikit-learn: To create a predictive model
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
# %% [markdown]
# ## 3. Data Wrangling
# %% [markdown]
# ### 3.1. Introduction to Datasets
# %% [markdown]
# #### 3.1.1. Election Data
# For election data, we'll be using data from the [MIT Election Data and Science Lab](https://electionlab.mit.edu/data) for 2020 election results. This lab focusing on collecting and analyzing election data to apply scientific research to the democracy of the United States.
# %%
elections = pd.read_csv("./data/1976-2020-president.csv")
# remove all other years except 2020
elections = elections[elections["year"] == 2020]
elections.head()
# %% [markdown]
# #### 3.1.2. Vaccination Data
# For COVID vaccine data we're going to use [the vaccination dataset](https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations) from [Our World in Data](https://ourworldindata.org/coronavirus). Our World in Data is an organization that focuses on gathering data and making it publicly available to facilitate research "[to make progress against the world's largest problems](https://ourworldindata.org/about#:~:text=Research%20and%20data%20to%20make%20progress%20against%20the%20world%E2%80%99s%20largest%20problems)." The data in this dataset comes directly from the CDC and the downloaded copy used for this project was last updated 5/15/21.
# %%
stateVaccines = pd.read_csv("./data/us_state_vaccinations.csv")
stateVaccines.info()
# %% [markdown]
# ### 3.2. Tidying the Data
# %% [markdown]
# There's some columns in the election dataset that we don't need, so we're going to get rid of those.
# %%
elections = elections[["state", "candidatevotes", "totalvotes", "party_simplified"]]
elections.head()
# %% [markdown]
# ## 4. Exploratory Data Analysis
# %% [markdown]
# ## 5. Hypothesis Testing and Machine Learning
# %% [markdown]
# ## 6. Conclusion
