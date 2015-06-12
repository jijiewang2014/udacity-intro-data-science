# -*- coding: utf-8 -*-

import numpy as np
import pandas
from ggplot import *
import scipy
import statsmodels.api as sm

"""
In this optional exercise, you should complete the function called 
predictions(turnstile_weather). This function takes in our pandas 
turnstile weather dataframe, and returns a set of predicted ridership values,
based on the other information in the dataframe.  

In exercise 3.5 we used Gradient Descent in order to compute the coefficients
theta used for the ridership prediction. Here you should attempt to implement 
another way of computing the coeffcients theta. You may also try using a reference implementation such as: 
http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html

One of the advantages of the statsmodels implementation is that it gives you
easy access to the values of the coefficients theta. This can help you infer relationships 
between variables in the dataset.

You may also experiment with polynomial terms as part of the input variables.  

The following links might be useful: 
http://en.wikipedia.org/wiki/Ordinary_least_squares
http://en.wikipedia.org/w/index.php?title=Linear_least_squares_(mathematics)
http://en.wikipedia.org/wiki/Polynomial_regression

This is your playground. Go wild!

How does your choice of linear regression compare to linear regression
with gradient descent computed in Exercise 3.5?

You can look at the information contained in the turnstile_weather dataframe below:
https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

Note: due to the memory and CPU limitation of our amazon EC2 instance, we will
give you a random subset (~10%) of the data contained in turnstile_data_master_with_weather.csv

If you receive a "server has encountered an error" message, that means you are hitting 
the 30 second limit that's placed on running your program. See if you can optimize your code so it
runs faster.
"""

def predictions(weather_turnstile):

    # Select Features (try different features!)
    features = weather_turnstile[['rain', 'precipi', 'Hour', 'mintempi', 'fog']]

    # Values
    values = weather_turnstile['ENTRIESn_hourly']

    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Call statsmodels
    features_array_constant = sm.add_constant(features_array)
    model = sm.OLS(values_array, features_array_constant)
    theta_statsmodels = model.fit().params

    # Compute prediction
    prediction = np.dot(features_array_constant, theta_statsmodels)

    # Provide some outputs
    print compute_r_squared(values_array, prediction)
    print values_array
    print prediction
    print weather_turnstile['precipi']

    # Plot
    plot_df = weather_turnstile[['precipi', 'ENTRIESn_hourly']]
    plot = ggplot(plot_df, aes('precipi', 'ENTRIESn_hourly')) + geom_point()
    print plot

    return prediction

def compute_r_squared(data, predictions):
    '''
    Given a list of original data points, and also a list of predicted data points,
    calculate the coefficient of determination (R^2) for this data.
    '''

    r_squared = 1 - np.square(data - predictions).sum() / np.square(data - data.mean()).sum()

    return r_squared

if __name__ == "__main__":
    df = pandas.read_csv('data/turnstile_data_master_with_weather.csv')
    oracle = predictions(df)
