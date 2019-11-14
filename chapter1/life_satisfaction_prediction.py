import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model

# load the data

oecd_bli = pd.read_csv('oecd_bli_2015.csv', thousands=',')

gdp_per_capita = pd.read_csv('gdp_per_capita.csv', thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')


def prepare_country_stats(oecd, gdp):
    oecd = oecd[oecd["INEQUALITY"] == "TOT"]
    oecd = oecd.pivot(index="Country", columns="Indicator", values="Value")
    gdp.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd, right=gdp,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]

# visualize the data
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for cyprus
X_new = [[22587]]  # Cyprus gdp per capita
print(model.predict(X_new))
