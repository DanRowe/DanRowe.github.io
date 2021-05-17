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
# - [us](https://github.com/unitedstates/python-us): Detailed state information
# %%
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import us
import sklearn
from sklearn.linear_model import LinearRegression
# %% [markdown]
# ## 3. Data Wrangling
# %% [markdown]
# ### 3.1. Introduction to Datasets
# %% [markdown]
# #### 3.1.1. Election Data
# For election data, we'll be using data from the [MIT Election Data and Science Lab](https://electionlab.mit.edu/data) for 2020 election results. This lab focusing on collecting and analyzing election data to apply scientific research to the democracy of the United States.
#
# MIT Election Data and Science Lab, 2017, "U.S. President 1976–2020", https://doi.org/10.7910/DVN/42MVDX, Harvard Dataverse, V6, UNF:6:4KoNz9KgTkXy0ZBxJ9ZkOw==
# %%
elections = pd.read_csv("./data/1976-2020-president.csv")
# remove all other years except 2020
elections = elections[elections["year"] == 2020]
elections.head()
# %% [markdown]
# #### 3.1.2. Vaccination Data
# For COVID vaccine data we're going to use [the vaccination dataset](https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations) from [Our World in Data](https://ourworldindata.org/coronavirus). Our World in Data is an organization that focuses on gathering data and making it publicly available to facilitate research "[to make progress against the world's largest problems](https://ourworldindata.org/about#:~:text=Research%20and%20data%20to%20make%20progress%20against%20the%20world%E2%80%99s%20largest%20problems)." The data in this dataset comes directly from the CDC and the downloaded copy used for this project was last updated 5/15/21.
#
# Mathieu, E., Ritchie, H., Ortiz-Ospina, E. et al. A global database of COVID-19 vaccinations. Nat Hum Behav (2021). https://doi.org/10.1038/s41562-021-01122-8
# %%
vaccinations = pd.read_csv("./data/us_state_vaccinations.csv")
vaccinations.info()
# %% [markdown]
# ### 3.2. Tidying the Data
# %% [markdown]
# There's some columns in the election dataset that we don't need, so we're going to get rid of those.
# %%
elections = pd.DataFrame(
    elections[["state", "candidatevotes", "totalvotes", "party_simplified"]])
elections.head()
# %% [markdown]
# Let's also calculate the party that had the most votes for each state to use later.
# %%
idx = elections.groupby(["state"])["candidatevotes"].transform(
    max) == elections["candidatevotes"]
majority = elections[idx][["state", "party_simplified"]].rename(
    columns={"party_simplified": "p"})
elections["majority_party"] = majority["p"]
elections.head()
# %% [markdown]
# Now we need to combine every state into one row so it's easier to use and rename the state column to location to match the vaccination dataset.
# %%
for i, row in elections.iterrows():
    elections.at[i, row["party_simplified"] +
                 "_percent"] = row["candidatevotes"]/row["totalvotes"]*100
elections = elections.groupby("state").first().drop(
    columns=["party_simplified", "candidatevotes"]).reset_index().rename(columns={"state": "location"})
elections.head()
# %% [markdown]
# For vaccination data, we need to remove federal entities like the *Dept of Defense* to ensure the dataset is only states. Also, make sure that dates are in datetime format.
# %%
states = [state.name for state in us.STATES]
states.append("New York State")
vaccinations = pd.DataFrame(
    vaccinations[vaccinations["location"].isin(states)])
vaccinations["date"] = pd.to_datetime(vaccinations["date"])
# %% [markdown]
# Now let's add the election data to the vaccination dataset.
# %%
for i, row in vaccinations.iterrows():
    location = row["location"].upper()
    if (location == "NEW YORK STATE"):
        location = "NEW YORK"
    stateElection = elections[elections["location"] == location]
    party = stateElection["majority_party"].iloc[0]
    vaccinations.at[i,
                    "DEMOCRAT_percent"] = stateElection["DEMOCRAT_percent"].iloc[0]
    vaccinations.at[i, "party"] = party
    color = "green"
    if (party == "REPUBLICAN"):
        color = "red"
    elif (party == "DEMOCRAT"):
        color = "blue"
    vaccinations.at[i, "color"] = color
colors = vaccinations[["location", "color"]].set_index("location").to_dict()[
    "color"]
vaccinations.head()
# %% [markdown]
# Next, we need to make a dataset that contains the most recent data entry for convenience.
# %%
idx = vaccinations.groupby("location")["date"].transform(
    max) == vaccinations["date"]
recentVax = vaccinations[idx]
recentVax.info()
# %% [markdown]
# ## 4. Exploratory Data Analysis
# Now that the data is easier to manipulate, lets explore the vaccination data and the relationships between different columns. We'll start by looking at the amount of people fully vaccinated in each state colored by political party.
# %%
recentVax = recentVax.sort_values(
    by=["people_fully_vaccinated"], ascending=False)
plt.figure(figsize=(8, 32))
sns.barplot(y=recentVax["location"],
            x=recentVax["people_fully_vaccinated"], palette=colors)
plt.xlabel("People Fully Vaccinated")
plt.ylabel("State")
plt.title("People Fully Vaccinated By State")
plt.show()
# %% [markdown]
# From the above chart it appears as if California, Texas, and New York State are way better at making sure people are fully vaccinated. However, this does not consider that states with a smaller population aren't going to have as many people available to vaccinate.
#
# Because of this, we will look at the amount of people fully vaccinated per 100 people in the total population. This comparison levels the playing field by standardizing the data on the total population.
# %%
recentVax = recentVax.sort_values(
    by=["people_fully_vaccinated_per_hundred"], ascending=False)
plt.figure(figsize=(8, 32))
sns.barplot(y=recentVax["location"],
            x=recentVax["people_fully_vaccinated_per_hundred"], palette=colors)
plt.xlabel("People Fully Vaccinated per 100 People")
plt.ylabel("State")
plt.title("People Fully Vaccinated Per 100 People By State")
plt.show()
# %% [markdown]
# From the above chart, we see a dramatically different picture than the first comparison. Here we can see that there's a more gradual difference between the the amount of people fully vaccinated in each state. More importantly, we can see that the leaders of states that are fully vaccinated have changed. Maine, Connecticut, Vermont, Massachusetts, and Rhode Island have the highest amount of people fully vaccinated per 100 people in their population. Also, Democratic states are leading the amount of people fully vaccinated with the closest Republican state being in 12th
# %% [markdown]
# Now, let's take a look at the rate at which each state administered the vaccination.
# %%
vaccinations.pivot(index="date", columns="location",
                   values="daily_vaccinations_per_million").plot(subplots=True, sharey=True, layout=(10, 5), figsize=(25, 25), color=colors)
plt.ylabel("People Vaccinated Per Million")
plt.xlabel("Day")
plt.show()
# %% [markdown]
# The above chart shows that a lot of the charts start off as an incline and then taper off about 75% of the way through. Some states like New Hampshire had bursts of increases and Maine appears to be increasing steadily. Let's pick some of the interesting charts and look at them a bit closer.
# %%
fig, ax = plt.subplots()
# Republican states
vaccinations.query("location == 'Florida'").plot(
    x="date", y="daily_vaccinations_per_million", ax=ax, linestyle="dashed", label="Florida")
vaccinations.query("location == 'Ohio'").plot(
    x="date", y="daily_vaccinations_per_million", ax=ax, linestyle="dashed", label="Ohio")
vaccinations.query("location == 'South Dakota'").plot(
    x="date", y="daily_vaccinations_per_million", ax=ax, linestyle="dashed", label="South Dakota")
# Democrat states
vaccinations.query("location == 'Maine'").plot(
    x="date", y="daily_vaccinations_per_million", ax=ax, label="Maine")
vaccinations.query("location == 'New Hampshire'").plot(
    x="date", y="daily_vaccinations_per_million", ax=ax, label="New Hampshire")
vaccinations.query("location == 'Vermont'").plot(
    x="date", y="daily_vaccinations_per_million", ax=ax, label="Vermont")
plt.ylabel("People Vaccinated Per Million")
plt.xlabel("Day")
plt.title("(Dashed = Republican)")
plt.suptitle("Republican vs Democratic States Daily Vaccinated")
plt.show()
# %% [markdown]
# The states chosen for the chart above were the states with the steepest incline for daily vaccination rate. The dashed lines on the chart represent states that voted primarily democratic. From this chart, we can observe that democratic states stay slightly above republican states in daily vaccination rates. However, the lines stay fairly even with each other until March where the democratic states began administering vaccinations at an increased rate.
# 
# Let's group them together to make this a little easier to see.
# %%
maxVax = vaccinations.groupby("location").agg(max)
plt.figure()
x = maxVax["people_fully_vaccinated_per_hundred"]
y = maxVax["DEMOCRAT_percent"]
sns.scatterplot(data=maxVax, x="people_fully_vaccinated_per_hundred",
                y="DEMOCRAT_percent", hue="party", palette=["red", "blue"])
# plt.scatter(x, y, alpha=0.7, c="Animation", colormap="jet")
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color="red")
plt.ylabel("Democratic Vote %")
plt.xlabel("People Fully Vaccinated per Hundred")
plt.title("Democratic Vote % vs People Fully Vaccinated in Each State")
plt.show()
# %% [markdown]
# From the chart above, it is clear that states that voted democratic are more likely to have a higher population of vaccinated people.
# %% [markdown]
# ## 6. Conclusion
# Looking at the charts and the data explored above, it is plausible that the vaccination rates are political. We saw a majority of democratic states having higher vaccination statistics than republican states. 

# %%
