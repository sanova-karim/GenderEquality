import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

import os
print(os.getcwd())

import warnings
warnings.filterwarnings("ignore")

# Load the CSV file
df = pd.read_csv('gender_data.csv')

# Extract the period column
year_column = df.columns[0]

# Prompt for country names
country_names = input("Enter the names of countries (comma-separated): ")
countries = [country.strip() for country in country_names.split(",")]

# Create a figure and axes for the plot
fig, ax = plt.subplots()

# Iterate over the selected countries
for country in countries:
    # Create the column names variable for GDI and GII
    gdi_column = f"{country} - GDI"
    gii_column = f"{country} - GII"

    # Check if columns exist
    if not gdi_column in df.columns or not gii_column in df.columns:
        print(f"Error: Data for {country} does not exist in the dataset. Exiting the program.")
        sys.exit(0)  # Stops the execution of the program

    # Extract the data for the current country
    country_data = df[[year_column, gdi_column, gii_column]]

    # Remove rows with missing data for the GDI column only
    country_data = country_data.dropna(subset=[gdi_column])

    # Convert the period column to datetime format
    country_data[year_column] = pd.to_datetime(country_data[year_column], format='%Y')

    # Set the period column as the index
    country_data.set_index(year_column, inplace=True)

    # Resample the data to annual frequency
    country_data = country_data.resample('AS').mean()

    # Plot the historical GDI data
    gdi_actual, = ax.plot(country_data.index, country_data[gdi_column], label=f'{country} - GDI')

    # Store the color used for plotting the actual GDI
    gdi_color = gdi_actual.get_color()

    # Perform grid search to find the optimal parameters for GDI
    best_gdi_aic = float('inf')

    for p in range (0,3, 1):
        for d in range (0,3,1):
            for q in range (0,3,1):
                current_model = ARIMA(endog=country_data[gdi_column], order=(p, d, q))
                current_fit = current_model.fit()
                current_aic = current_fit.aic

        # Update best parameters if the current model has lower AIC
                if current_aic < best_gdi_aic:
                    best_gdi_aic = current_aic
                    gdi_model_fit = current_fit

    # Make predictions for GDI
    gdi_predictions = gdi_model_fit.predict(start=len(country_data), end=len(country_data) + 20)

    # Plot the predicted GDI data
    ax.plot(gdi_predictions.index, gdi_predictions, linestyle='dashed', color=gdi_color)

    # Plot the historical GII data
    gii_actual, = ax.plot(country_data.index, country_data[gii_column], label=f'{country} - GII')

    # Store the color used for plotting the actual GII
    gii_color = gii_actual.get_color()

    # Perform grid search to find the optimal parameters for GII
    best_gii_aic = float('inf')

    for p in range (0,3, 1):
        for d in range (0,3,1):
            for q in range (0,3,1):
                current_model = ARIMA(endog=country_data[gii_column], order=(p, d, q))
                current_fit = current_model.fit()
                current_aic = current_fit.aic

        # Update best parameters if the current model has lower AIC
                if current_aic < best_gii_aic:
                    best_gii_aic = current_aic
                    gii_model_fit = current_fit

    # Make predictions for GII
    gii_predictions = gii_model_fit.predict(start=len(country_data), end=len(country_data) + 20)

    # Plot the predicted GII data
    ax.plot(gii_predictions.index, gii_predictions, linestyle='dashed', color=gii_color)

# Show the legend for each country in separate rows
ax.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=2)

# Set the plot title and labels
ax.set_title('Gender Development and Inequality Indices (GDI & GII)')
ax.set_xlabel('Year')
ax.set_ylabel('GDI/GII')

# Add a vertical line at year 2013
ax.axvline(pd.to_datetime('2013'), color='black', linestyle='--')

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.show()



