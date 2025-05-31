# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:15:16 2025

@author: Abhishek Singh
"""

# Exploratory Data Analysis on data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates

# Load the CSV data
df = pd.read_csv('Rainfall_data.csv')

# Create a datetime column and set as index
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df.set_index('Date', inplace=True)

# Drop the now redundant Year, Month, Day columns if desired
df.drop(columns=['Year', 'Month', 'Day'], inplace=True)

# Display basic summary statistics
print(df.describe())

# Plot the time series for precipitation
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Precipitation'], label='Precipitation', color='royalblue')
plt.title('Monthly Precipitation (2000-2020)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Place a tick every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format labels as Year
plt.xticks(rotation=45)  # Rotate for better readability
plt.show()


# Seasonal boxplot for Precipitation (group by month)
df['Month'] = df.index.month
df['Month_Name'] = df.index.strftime('%b')
plt.figure(figsize=(10, 6))
sns.boxplot(x='Month_Name', y='Precipitation', data=df)
plt.title('Monthly Distribution of Precipitation')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')
plt.show()

# Pair Plot for Climate Variables
sns.pairplot(df[['Specific Humidity', 'Relative Humidity', 'Temperature', 'Precipitation']],
             diag_kind='kde',
             plot_kws={'alpha': 0.5, 's': 40, 'edgecolor': 'k'})
plt.suptitle("Pair Plot of Climate Variables", y=1.02)
plt.show()

# Rolling Average of Precipitation
df['Precip_Rolling'] = df['Precipitation'].rolling(window=12).mean()
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Precipitation'], alpha=0.4, label='Monthly Precipitation')
plt.plot(df.index, df['Precip_Rolling'], color='red', linewidth=2, label='12-Month Rolling Average')
plt.title("Precipitation with 12-Month Rolling Average")
plt.xlabel("Year")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Place a tick every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format labels as Year
plt.xticks(rotation=45)  # Rotate for better readability
plt.show()


# Decomposition of precipitation time series (ensure frequency is set correctly)
decomposition = seasonal_decompose(df['Precipitation'], period=12, model='additive')
observed = decomposition.observed
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Common formatter and locator
year_locator = mdates.YearLocator()
year_formatter = mdates.DateFormatter('%Y')

# Observed
axs[0].plot(observed, label='Observed', color='blue')
axs[0].set_ylabel('Rainfall (mm)')
axs[0].set_title('Observed Precipitation')

# Trend
axs[1].plot(trend, label='Trend', color='green')
axs[1].set_ylabel('Rainfall (mm)')
axs[1].set_title('Trend Component')

# Seasonal
axs[2].plot(seasonal, label='Seasonal', color='orange')
axs[2].set_ylabel('Rainfall (mm)')
axs[2].set_title('Seasonal Component')

# Residual
axs[3].plot(residual, label='Residual', color='gray')
axs[3].set_ylabel('Rainfall (mm)')
axs[3].set_xlabel('Year')
axs[3].set_title('Residual Component')

# Format x-axis for all
for ax in axs:
    ax.xaxis.set_major_locator(year_locator)
    ax.xaxis.set_major_formatter(year_formatter)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Seasonal Decomposition of Monthly Precipitation (Additive Model)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# Correlation Matrix Heatmap
corr_matrix = df[['Specific Humidity', 'Relative Humidity', 'Temperature', 'Precipitation']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Climate Variables")
plt.show()


# Check stationarity using the ADF test
result = adfuller(df['Precipitation'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
