1. This program (GnderDevIneq.py) predicts gender inquaility for a given country(or countries) using two indices - Gender Development Index (GDI) and Gender Inequality Index (GII)
   Both are measures of gender inequality. It is better to have high GDI (max 1.0) and low GII (min 0.0).
2. Historical data used from IMF (International Monetary Fund). It is named as gender_data.csv
3. The program reads the data, plots the actual GDI and GII data for a country (or comma separated countries), and plots the forecast for the next 20 years
4. Machine learning model ARIMA was used to predict data until 2033. ARIMA is a class of statistical models for analyzing and forecasting time series data.
5. To run the program simply type in the command prompt: py GenderDevIneq.py 
