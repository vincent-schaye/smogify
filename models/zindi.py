import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform

from preprocessing import DateTransformer, InterpolationTransformer, create_preprocessor, drop_rows_with_missing_values

##########################################################

# Define your parameters and assumptions
threshold = 0  # Maximum number of missing values (i.e. columns) allowed in a row - used in Interpolater preprocessor
columns_to_drop = ['target_min', 'target_max', 'target_variance', 'target_count', 'L3_CH4_CH4_column_volume_mixing_ratio_dry_air', 'L3_CH4_aerosol_height', 'L3_CH4_aerosol_optical_depth', 'L3_CH4_sensor_azimuth_angle', 'L3_CH4_sensor_zenith_angle', 'L3_CH4_solar_azimuth_angle', 'L3_CH4_solar_zenith_angle']
                    # Drop the target stuff. And also the last 7 columns which were BASICALLY empty ...
columns_to_keep = [] # not used
target_column = 'target'
RSEED = 42

##########################################################

# define function to execute everything
def score(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n{model_name} Performance:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    # Check for overfitting/underfitting
    if train_rmse < test_rmse * 0.8:
        print("The model might be overfitting.")
    elif train_rmse > test_rmse * 1.2:
        print("The model might be underfitting.")
    else:
        print("The model seems to be fitting well.")

##########################################################

# define function to execute for final Zindi submission
def score_zindi(model, base_pipeline, test_df, output_file='Zindi_submission.csv'):
    # Process the test data using the base pipeline
    X_test_processed = base_pipeline.transform(test_df)

    # Make predictions
    predictions = model.predict(X_test_processed)

    # Create the submission dataframe
    submission_df = pd.DataFrame({
        'Place_ID X Date': test_df['Place_ID X Date'],
        'target': predictions
    })

    # Save the submission to a CSV file
    submission_df.to_csv(output_file, index=False)

    print(f"Zindi submission file created: {output_file}")

    return predictions, submission_df

##########################################################

#Load data
data = pd.read_csv('../data/Train.csv') # might need to modify later, if Okan creates new master list

# Define your feature groups
# Think to see if breaks...I already included Date_DT here before date_transform pipeline. Also how to handle weekday/weekend stuff
num_features = ['precipitable_water_entire_atmosphere',
       'relative_humidity_2m_above_ground',
       'specific_humidity_2m_above_ground', 'temperature_2m_above_ground',
       'u_component_of_wind_10m_above_ground',
       'v_component_of_wind_10m_above_ground',
       'L3_NO2_NO2_column_number_density',
       'L3_NO2_NO2_slant_column_number_density',
       'L3_NO2_absorbing_aerosol_index', 'L3_NO2_cloud_fraction',
       'L3_NO2_stratospheric_NO2_column_number_density',
       'L3_NO2_tropopause_pressure', 'L3_O3_O3_column_number_density',
       'L3_O3_O3_effective_temperature', 'L3_O3_cloud_fraction',
       'L3_CO_CO_column_number_density', 'L3_CO_H2O_column_number_density',
       'L3_CO_cloud_height', 'L3_HCHO_HCHO_slant_column_number_density',
       'L3_HCHO_cloud_fraction',
       'L3_HCHO_tropospheric_HCHO_column_number_density',
       'L3_HCHO_tropospheric_HCHO_column_number_density_amf',
       'L3_CLOUD_cloud_base_height', 'L3_CLOUD_cloud_base_pressure',
       'L3_CLOUD_cloud_fraction', 'L3_CLOUD_cloud_optical_depth',
       'L3_CLOUD_cloud_top_height', 'L3_CLOUD_cloud_top_pressure',
       'L3_CLOUD_surface_albedo', 'L3_AER_AI_absorbing_aerosol_index',
       'L3_AER_AI_sensor_altitude', 'L3_SO2_SO2_column_number_density',
       'L3_SO2_SO2_column_number_density_amf',
       'L3_SO2_SO2_slant_column_number_density',
       'L3_SO2_absorbing_aerosol_index', 'L3_SO2_cloud_fraction']
cat_features = ['Place_ID', 'Date_DT','Place_ID X Date']

# Train-test-split up front!
X = data.drop(target_column, axis=1)
y = data[target_column]

# Drop rows with too many missing values before splitting the data
X, y = drop_rows_with_missing_values(X, y,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RSEED)

# Create the preprocessor
preprocessor = create_preprocessor(num_features, cat_features)

# Define the base pipeline
base_pipeline = Pipeline([
    ('date_transform', DateTransformer()),
    #('advanced_imputer', AdvancedTimeSeriesImputer(method='pmdarima', seasonal_period=12)),
    ('interpolator', InterpolationTransformer(columns_to_drop=columns_to_drop)),
    ('preprocessor', preprocessor)
])


# Fit the base pipeline on training data and transform both train and test data
X_train_processed = base_pipeline.fit_transform(X_train)
X_test_processed = base_pipeline.transform(X_test)

# Create and train the XGBoost model with the best parameters found
best_model = XGBRegressor(
    learning_rate=0.1318390965394111,
    max_depth=7,
    n_estimators=85,
    random_state=42
)
best_model.fit(X_train_processed, y_train)

##########################################################

# Execute for Zindi

# After training and selecting your best model
# best_model = grid_search.best_estimator_  # this might take too long again so re-think this!!!!!

# Load the test data for Zindi submission
zindi_test_data = pd.read_csv('../data/Test.csv')  # Save the Zindi test.csv into new name/ variable

# Create the Zindi submission
zindi_predictions, zindi_submission_df = score_zindi(best_model, base_pipeline, zindi_test_data) # this will create/ save new csv file in folder too

print(zindi_submission_df)