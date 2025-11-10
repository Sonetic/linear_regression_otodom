# linear_regression_otodom


**Project Overview**

This project predicts property prices in Warsaw using different linear regression approaches. Main steps include:

**clean_data.py** - cleans raw real estate listings, converts prices to PLN, extracts surface area and coordinates, and saves a cleaned CSV.

**columns_prep.py** - adds district information, cleans numeric columns, filters extreme values, and creates dummy variables for categorical features.

**linear_regression_analytical.py** - predicts prices analytically by computing the exact regression coefficients using matrix operations.

**linear_regression_libraries.py** - uses sklearn LinearRegression with PolynomialFeatures to capture interactions between numeric and categorical features.

**linear_regression_gradient_descent.py** - trains a linear regression model using gradient descent and visualizes the cost function over iterations.

**test.py** - evaluates all three models on the dataset, printing MAE, RMSE, and R² for comparison.

**test_site.py** - runs a Flask web app showing predictions for sample properties and displaying regression metrics (templates/results.html).




#
#
#
**clean_data.py**

The script processes real estate listings data and prepares it for analysis. The main steps include:

- Reading raw data: Loads CSV files containing JSON-encoded listings.
- JSON unpacking: Converts JSON strings into structured DataFrames.
- Price cleaning: Cleans and converts prices to PLN (supports €, $, PLN).
- Surface cleaning: Converts surface area values to numeric format.
- Coordinate extraction: Extracts Longitude and Latitude from the location field.
- Saving cleaned data: Outputs the cleaned dataset as a new CSV file ready for analysis.




**columns_prep.py**
The script prepares the cleaned real estate dataset for modeling by adding features and filtering out invalid entries. 
Main steps:
- Load cleaned data: Reads otodom_cleaned.csv.
- Define district boundaries: Estimates boundaries of Warsaw districts by latitude and longitude.
- Assign districts: Adds a district column based on coordinates; removes listings outside Warsaw.
- Clean numeric columns: Converts no_of_rooms, surface_num, and price_num to numeric, drops rows with missing values.
- Filter extreme values: Removes listings with prices or surface areas outside reasonable ranges.
- Create dummy variables: Converts categorical district column into numeric dummy columns (number of districts) for modeling. Each dummy column represents a district with 1 if the listing belongs to that district
  and 0 otherwise.
- Drop first dummy column: Drops the first district column to avoid multicollinearity in regression models (so one district becomes the reference category).
- Set column order for modeling: Ensures columns are in the correct order and adds a bias term.




**linear_regression_analitycal.py**

The script redicts real estate prices using an analytical linear regression approach. 
Main steps:
- Load prepared data: Copies the dataset from columns_prep.py.
- Create interaction features: Generates interactions between district dummy variables and numeric variables (surface_num and no_of_rooms).
- Select features: Prepares feature matrix X and target vector y.
- Normalize numeric features: Standardizes numeric columns to improve regression stability.
- Compute coefficients analytically: Uses the normal equation to calculate regression coefficients (theta).
- Prepare test vector: Adds interaction features and normalizes numeric values for the input test case.
- Predict price: Computes the predicted price as the dot product of test vector and coefficients.
- In file: Tests model for specific parameters.


**linear_regression_gradient_descent.py**
The script predicts real estate prices using linear regression trained with gradient descent. 
Main steps:
- Load prepared data: Copies the dataset from columns_prep.py.
- Create interaction features: Generates interactions between district dummy variables and numeric variables (surface_num and no_of_rooms).
- Normalize numeric features: Standardizes numeric columns for stable gradient descent.
- Train model with gradient descent: Iteratively updates coefficients to minimize mean squared error (MSE).
- Plot training error: Visualizes the change in MSE over iterations.
- Predict price: Computes the predicted price for a given test input using the learned coefficients.
- In file: Tests model for specific parameters and shows the plot of MSE.



**linear_regression_libraries.py**
Predicts real estate prices using Scikit-learn’s LinearRegression with polynomial interaction features. 
Main steps:
- Load prepared data: Copies the dataset from columns_prep.py.
- Select base features: Uses surface_num, no_of_rooms, and district dummy variables.
- Generate interaction features: Uses PolynomialFeatures(interaction_only=True) to create interactions automatically between base features.
- Normalize numeric features: Standardizes numeric columns for stable training.
- Train model: Fits a linear regression model on the transformed features.
- Predict price: Prepares a test input, applies the same transformations, and computes the predicted price.
- In file: Tests model for specific parameters.

Note: Using PolynomialFeatures (automatic interactions) instead of manual interactions in linear_regression_analitycal increases model flexibility and improves R² by about +3 percentage points.



**test.py**
Compares predictions of all linear regression models on real estate data and evaluates their performance. 
Main steps:
- Load data: Copies the cleaned dataset from columns_prep.py.
- Prepare inputs: Creates feature dictionaries for each sample.
- Generate predictions: Uses Gradient Descent, Analytical, and Scikit-learn models.
- Evaluate metrics: Calculates MAE, MSE, RMSE, and R² for each model.
- Display results: Prints sample predictions and regression metrics.



**test_site_py**
Runs a Flask web application to display predicted real estate prices and model performance. 
Main steps:
- Load dataset: Copies cleaned data from columns_prep.py.
- Prepare inputs: Builds feature dictionaries for all properties.
- Generate predictions: Uses Gradient Descent, Analytical, and Sklearn LinearRegression models.
- Compute metrics: Calculates MAE, RMSE, and R² for each model.
- Display results: Renders an HTML page showing actual vs predicted prices and performance metrics using the templates/results.html template.




**Model Comparison**


Model                   MAE	        RMSE	    R²

Gradient Descent	242,864.82 | 360,943.38 | 0.7003

Analytical	        242,217.91 |	359,656.35 |	0.7024

Sklearn	            232,881.30 |	340,883.54 |	0.7326





#
#
Summary and Evaluation

All three models successfully predict property prices in Warsaw using the cleaned dataset.
Sklearn LinearRegression with PolynomialFeatures is the most accurate, achieving R² = 0.7326, which shows it captures interactions between features effectively.
Analytical regression provides reliable results (R² = 0.7024) and closely matches the gradient descent approach.
Gradient Descent converges well and produces reasonable predictions (R² = 0.7003), though slightly less accurate than analytical and sklearn models.

The predictions are not perfect because the models only consider three variables (surface, number of rooms, and district). Real estate prices often follow complex, nonlinear patterns, 
so linear regression may not capture all factors influencing the market.

Conclusion:
The project meets its objectives: cleaning and preparing data, implementing multiple regression methods, and comparing their performance.
The models perform as expected, the workflow is reproducible, and the web interface successfully displays predictions and metrics. Overall, the project is a success,
with room for future improvements by including more features or using nonlinear models.