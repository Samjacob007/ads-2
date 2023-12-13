# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:11:11 2023

@author: sam jacob
"""

#!/usr/bin/env python
# coding: utf-8

"""
This script performs data analysis and visualization on carbon-related data for specific countries over the years.
It includes data loading, transformation, cleaning, statistical analysis, correlation analysis, and visualization.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis

# Load the carbon-related dataset
carbon = pd.read_csv("C:/Users/sam jacob/OneDrive/Desktop/ads 2/dataset/apico2_csv.csv")
print(carbon)

"""
Define Feature codes and country names for analysis.
"""

COLUMN_CODES = ['Country Name', 'Country Code', 'Year','EN.ATM.CO2E.GF.ZS','EN.ATM.CO2E.GF.KT',
                'EN.ATM.CO2E.EG.ZS','EG.USE.PCAP.KG.OE','EG.USE.ELEC.KH.PC','EG.ELC.PETR.ZS',
                'EG.ELC.NUCL.ZS','EG.ELC.NGAS.ZS','EG.ELC.HYRO.ZS','EG.ELC.COAL.ZS']

featureMap = {
    "EN.ATM.CO2E.GF.ZS": "CO2 emissions from gaseous fuel consumption (% of total)",
    "EN.ATM.CO2E.GF.KT": "CO2 emissions from gaseous fuel consumption (kt)",
    "EN.ATM.CO2E.EG.ZS": "CO2 intensity (kg per kg of oil equivalent energy use)",
    "EG.USE.PCAP.KG.OE": "Energy use (kg of oil equivalent per capita)",
    "EG.USE.ELEC.KH.PC": "Electric power consumption (kWh per capita)",
    "EG.ELC.PETR.ZS": "Electricity production from oil sources (% of total)",
    "EG.ELC.NUCL.ZS": "Electricity production from nuclear sources (% of total)",
    "EG.ELC.NGAS.ZS": "Electricity production from natural gas sources (% of total)",
    "EG.ELC.HYRO.ZS": "Electricity production from hydroelectric sources (% of total)",
    "EG.ELC.COAL.ZS": "Electricity production from coal sources (% of total)",
}

countryMap = { 
    "AUS": "Australia",
    "CAN": "Canada",
    "DEN": "Denmark",
    "ESP": "Spain",
    "FI": "Finland",
    "FRA": "France",
    "UK": "United Kingdom"
}

# Melting the dataset
melted_df = carbon.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Value')

# Pivot the table to have Indicator Names as columns
pivoted_df = melted_df.pivot_table(index=['Country Name', 'Country Code', 'Year'], columns='Indicator Code', values='Value').reset_index()
pivoted_df.to_csv('C:/Users/sam jacob/OneDrive/Desktop/ads 2/dataset/pivoted1.csv')

# Filter columns for further analysis
filtered_columns = [col for col in pivoted_df.columns if col in COLUMN_CODES]
df_filtered = pivoted_df[filtered_columns]

# Fill NaN values with column means
df_cleaned = df_filtered.fillna(df_filtered.mean(numeric_only=True))
df_cleaned.to_csv('C:/Users/sam jacob/OneDrive/Desktop/ads 2/dataset/Cleaneddataset2.csv')

"""
Create DataFrames for specific countries (United Kingdom, Denmark, Finland).
"""
df_UK = df_cleaned[df_cleaned["Country Name"] == "United Kingdom"]
df_DEN = df_cleaned[df_cleaned["Country Name"] == "Denmark"]
df_FI = df_cleaned[df_cleaned["Country Name"] == "Finland"]
"""
Applying Statistical Methods on cleaned dataset.
"""
copy_df_cleaned = df_cleaned.drop(['Year', 'Country Name'], axis='columns')
print(copy_df_cleaned.describe())

skewness = skew(df_UK["EG.ELC.COAL.ZS"])
print(skewness)

kurtosis_ = skew(df_UK["EG.ELC.COAL.ZS"])
print(kurtosis_)



"""
Correlation Matrix and Heat map for United Kingdom.
"""
correlation_matrix_UK = df_UK.corr(numeric_only=True)
correlation_matrix_UK = correlation_matrix_UK.rename(columns=featureMap)
correlation_matrix_UK = correlation_matrix_UK.rename(index=featureMap)
plt.figure(2, figsize=(10, 10))
heatmap_data_UK = sns.heatmap(correlation_matrix_UK, annot=True, fmt=".1g", vmax=1, vmin=0)
plt.title('Correlation Matrix for United Kingdom')
plt.show()

"""
Correlation Matrix and Heat map for Finland.
"""
correlation_matrix_FI = df_FI.corr(numeric_only=True)
correlation_matrix_FI = correlation_matrix_FI.rename(columns=featureMap)
correlation_matrix_FI = correlation_matrix_FI.rename(index=featureMap)
plt.figure(2, figsize=(10, 10))
heatmap_data_FI = sns.heatmap(correlation_matrix_FI, annot=True, fmt=".1g", vmax=1, vmin=0)

plt.title('Correlation Matrix for Finland')
plt.show()

"""
Correlation Matrix and Heat map for Denmark.
"""
correlation_matrix_DEN = df_DEN.corr(numeric_only=True)
correlation_matrix_DEN = correlation_matrix_DEN.rename(columns=featureMap)
correlation_matrix_DEN = correlation_matrix_DEN.rename(index=featureMap)
plt.figure(2, figsize=(10, 10))
heatmap_data_DEN = sns.heatmap(correlation_matrix_DEN, annot=True, fmt=".1g", vmax=1, vmin=0)

plt.title('Correlation Matrix for Denmark')
plt.show()

"""
Filter and plot data for electricity production from natural gas sources.
"""
df_population_by_year = pd.read_csv("C:/Users/sam jacob/OneDrive/Desktop/ads 2/dataset/Cleaneddataset2.csv")
filtered_population = df_population_by_year[((df_population_by_year['Country Name'] == 'Denmark') |
                                            (df_population_by_year['Country Name'] == 'Spain') |
                                            (df_population_by_year['Country Name'] == 'United Kingdom') |
                                            (df_population_by_year['Country Name'] == 'Finland')) &
                                           ((df_population_by_year['Year'] == 1990) |
                                            (df_population_by_year['Year'] == 2000) |
                                            (df_population_by_year['Year'] == 2010))]
filtered_population = filtered_population[["Country Name", "Year", "EG.ELC.NGAS.ZS"]]
pivoted_population_df = filtered_population.pivot(index='Country Name', columns='Year', values='EG.ELC.NGAS.ZS').reset_index()
pivoted_population_df.plot(kind='bar', x='Country Name', y=[1990, 2000, 2010])
plt.xticks(rotation=0, horizontalalignment="center")
plt.title('Electricity production from natural gas sources (% of total) of country')
plt.xlabel('Country Name')
plt.ylabel('Electricity production from natural gas sources (% of total)')
plt.legend()
plt.grid(True)
plt.show()

"""
Filter and plot data for CO2 emissions from gaseous fuel consumption.
"""
filtered_population = df_population_by_year[((df_population_by_year['Country Name'] == 'Denmark') |
                                            (df_population_by_year['Country Name'] == 'Spain') |
                                            (df_population_by_year['Country Name'] == 'United Kingdom') |
                                            (df_population_by_year['Country Name'] == 'Finland')) &
                                           ((df_population_by_year['Year'] == 1990) |
                                            (df_population_by_year['Year'] == 2000) |
                                            (df_population_by_year['Year'] == 2010))]
filtered_population = filtered_population[["Country Name", "Year", "EN.ATM.CO2E.GF.ZS"]]
pivoted_population_df = filtered_population.pivot(index='Country Name', columns='Year', values='EN.ATM.CO2E.GF.ZS').reset_index()
pivoted_population_df.plot(kind='bar', x='Country Name', y=[1990, 2000, 2010])
plt.xticks(rotation=0, horizontalalignment="center")
plt.title('CO2 emissions from gaseous fuel consumption (% of total)')
plt.xlabel('Country Name')
plt.ylabel('CO2 emissions from gaseous fuel consumption (% of total)')
plt.legend()
plt.grid(True)
plt.show()

def plot_co2(file):
    """
    Plots line charts for CO2 emissions from gaseous fuel consumption and electricity production from nuclear sources.

    Parameters:
    file (str): Path to the CSV file containing the cleaned dataset.
    """
    data = pd.read_csv(file)
    countries = ['France', 'Denmark', 'United Kingdom', 'Finland', 'Spain']
    years = list(range(1990, 2000))

    filtered_data = data[(data['Country Name'].isin(countries)) & (data['Year'].isin(years))]
    pivoted_data = filtered_data.pivot(index='Year', columns='Country Name', values='EN.ATM.CO2E.GF.KT')

    plt.figure(figsize=(10, 6))
    for country in countries:
        plt.plot(pivoted_data.index, pivoted_data[country], label=country)

    plt.title('CO2 emissions from gaseous fuel consumption (kt) of country')
    plt.xlabel('Year')
    plt.ylabel('CO2 emissions from gaseous fuel consumption (kt)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function and pass the CSV file path as an argument
plot_co2("C:/Users/sam jacob/OneDrive/Desktop/ads 2/dataset/Cleaneddataset2.csv")



def plot_nuclear_production(file, countries, years):
    """
    Plots electricity production from nuclear sources for specified countries and years.

    Parameters:
    - file (str): Path to the CSV file containing the dataset.
    - countries (list): List of country names for which to plot data.
    - years (list): List of years for which to plot data.

    Returns:
    None
    """
    # Load the dataset from the CSV file
    data = pd.read_csv(file)

    # Filter data for the specified countries and years
    filtered_data = data[(data['Country Name'].isin(countries)) & (data['Year'].isin(years))]

    # Pivot the data for easier plotting
    pivoted_data = filtered_data.pivot(index='Year', columns='Country Name', values='EG.ELC.NUCL.ZS')
    
    # Plotting
    plt.figure(figsize=(10, 6))

    for country in countries:
        plt.plot(pivoted_data.index, pivoted_data[country], label=country)

    plt.title('Electricity Production from Nuclear Sources (% of Total) - Selected Countries')
    plt.xlabel('Year')
    plt.ylabel('Electricity Production from Nuclear Sources (% of Total)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Place legend outside the plot
    plt.grid(True)
    plt.show()

# Example usage
plot_nuclear_production("C:/Users/sam jacob/OneDrive/Desktop/ads 2/dataset/Cleaneddataset2.csv", 
                        ['France', 'Denmark', 'United Kingdom', 'Finland', 'Spain'],
                        list(range(1990, 2000)))

