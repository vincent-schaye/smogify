import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

from preprocessing import DateTransformer, InterpolationTransformer, create_preprocessor, \
    drop_rows_with_missing_values  # , AdvancedTimeSeriesImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet

##########################################################

# Define your parameters and assumptions
threshold = 60  # Maximum number of missing values (i.e. columns) allowed in a row - used in Interpolater preprocessor
columns_to_drop = ['target_min', 'target_max', 'target_variance', 'target_count',
                   'L3_CH4_CH4_column_volume_mixing_ratio_dry_air', 'L3_CH4_aerosol_height',
                   'L3_CH4_aerosol_optical_depth', 'L3_CH4_sensor_azimuth_angle', 'L3_CH4_sensor_zenith_angle',
                   'L3_CH4_solar_azimuth_angle', 'L3_CH4_solar_zenith_angle']
# Drop the target stuff. And also the last 7 columns which were BASICALLY empty ...
columns_to_keep = []  # not used
target_column = 'target'
RSEED = 42


##########################################################

# define function to execute everything
def score(model, X_train, X_test, y_train, y_test, model_name):
    # Check if the model supports early stopping (e.g., XGBoost)
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

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    print(f"\n{model_name} Performance:")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")

    # Check for overfitting/underfitting
    if train_rmse < test_rmse * 0.8:
        print("The model might be overfitting.")
    elif train_rmse > test_rmse * 1.2:
        print("The model might be underfitting.")
    else:
        print("The model seems to be fitting well.")


##########################################################
##########################################################

# Execute for own data sets

# Load data
data = pd.read_csv('../data/Train.csv')  # might need to modify later, if Okan creates new master list

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
cat_features = ['Place_ID', 'Date_DT', 'Place_ID X Date']

# Train-test-split up front!
X = data.drop(target_column, axis=1)
y = data[target_column]

# Drop rows with too many missing values before splitting the data
X, y = drop_rows_with_missing_values(X, y, threshold=threshold)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RSEED)

# Create the preprocessor
preprocessor = create_preprocessor(num_features, cat_features)

# Define the base pipeline
base_pipeline = Pipeline([
    ('date_transform', DateTransformer()),
    # ('advanced_imputer', AdvancedTimeSeriesImputer(method='pmdarima', seasonal_period=12)),
    ('interpolator', InterpolationTransformer(columns_to_drop=columns_to_drop)),
    ('preprocessor', preprocessor)
])

# Fit the base pipeline on training data and transform both train and test data
X_train_processed = base_pipeline.fit_transform(X_train)
X_test_processed = base_pipeline.transform(X_test)

# Create individual model pipelines
lr_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2)),
    ('elasticnet', ElasticNet())  # ElasticNet combines Lasso and Ridge
])

knn_model = Pipeline([
    ('scaler', MaxAbsScaler()),  # Use MaxAbsScaler for sparse matrices,  # Set with_mean=False for sparse matrices. Changed to MinMax Scaler after Standard led to R2 = 1.0 ...
    ('knn', KNeighborsRegressor(n_neighbors=5, metric='manhattan', weights='uniform'))  # KNN model. Changed weights to 'uniform'. 'distance' led to R2 = 1.0 clear overfitting
])

rf_model = RandomForestRegressor(
    min_samples_leaf=3,  # Experiment with different values
    max_depth=5,  # Control complexity
    n_estimators=50, random_state=42)

xgb_model = XGBRegressor(
    min_child_weight=3,# Increase to make the model more conservative
    learning_rate=0.05,  # Decrease for more conservative learning
    n_estimators=100,  # Increase to compensate for lower learning rate
    # early_stopping_rounds=10,  # Add early stopping - doesnt seem to work. Ask Arjun
    random_state=42)

#Declaring Parameters for testing

# Linear Regression with ElasticNet
lr_param_distributions = {
    'elasticnet__alpha': uniform(0.01, 20),  # Alpha for ElasticNet. Previous round with 0.001, 10 got R2 Test 0.3131
    'elasticnet__l1_ratio': uniform(0, 1)  # L1 ratio for ElasticNet, 1 is lasso and 0 is ridge
}

# K-Nearest Neighbors
knn_param_distributions = {
    'knn__n_neighbors': randint(5, 25),
    'knn__weights': ['uniform', 'distance'],  # Weights in KNN
    'knn__metric': ['manhattan']  # Distance metrics. Lets try manhattan only. Prev also euclidean
}

# Random Forest
rf_param_distributions = {
    'n_estimators': randint(25, 100), # was 25 to 100 before. Fit well
    'max_depth': randint(3, 10),
    'min_samples_leaf': randint(1, 5)
}

# XGBoost
xgb_param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(2, 6),
    'learning_rate': uniform(0.01, 0.1),
    'min_child_weight': randint(5, 10)
}


# Create RandomizedSearchCV for each model
def perform_random_search(model, param_distributions, X_train, y_train, n_iter=20, cv=5):
    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search

# Perform RandomizedSearchCV
best_models = {}
for name, (model, param_distributions) in {
    'Linear Regression': (lr_model, lr_param_distributions),
    'KNN': (knn_model, knn_param_distributions),
    'Random Forest': (rf_model, rf_param_distributions),
    'XGBoost': (xgb_model, xgb_param_distributions)
}.items():
    print(f"Performing RandomizedSearchCV for {name}...")
    search = perform_random_search(model, param_distributions, X_train_processed, y_train)
    best_model = search.best_estimator_
    best_models[name] = best_model

    # Calculate RMSE and R²
    best_rmse = np.sqrt(-search.best_score_)
    y_test_pred = best_model.predict(X_test_processed)
    r2 = r2_score(y_test, y_test_pred)

    print(f"Best parameters for {name}: {search.best_params_}")
    print(f"Best RMSE for {name}: {best_rmse:.4f}")
    print(f"R² score for {name}: {r2:.4f}\n")

# Score the best models directly
for name, model in best_models.items():
    score(model, X_train_processed, X_test_processed, y_train, y_test, f"{name} (Best Model)")

# Use the best models found for stacking
stacking_model = StackingRegressor(
    estimators=[
        ('lr', best_models['Linear Regression']),
        ('knn', best_models['KNN']),
        ('rf', best_models['Random Forest']),
        ('xgb', best_models['XGBoost'])
    ],
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42)
)

# Perform RandomizedSearchCV for stacking model
stacking_param_distributions = {
    'final_estimator__n_estimators': randint(50, 150),
    'final_estimator__max_depth': randint(3, 8)
}

stacking_random_search = RandomizedSearchCV(
    stacking_model,
    stacking_param_distributions,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1, # means use all available cores
    random_state=42
)

stacking_random_search.fit(X_train_processed, y_train)
print(f"\nRandom Search Results:")
best_stacking_model = stacking_random_search.best_estimator_
score(best_stacking_model, X_train_processed, X_test_processed, y_train, y_test, "Best Stacking Model (After RandomizedSearchCV)")

# Perform RandomizedSearchCV for voting model
# Create the Voting Regressor
voting_model = VotingRegressor(estimators=[
    ('lr', best_models['Linear Regression']),
    ('knn', best_models['KNN']),
    ('rf', best_models['Random Forest']),
    ('xgb', best_models['XGBoost'])
])

# Fit the Voting Regressor on the training data
voting_model.fit(X_train_processed, y_train)

# Evaluate the model on the test data
score(voting_model, X_train_processed, X_test_processed, y_train, y_test, "Voting Model")


"""
# Score individual models
score(lr_model, X_train_processed, X_test_processed, y_train, y_test, "Linear Regression")
score(knn_model, X_train_processed, X_test_processed, y_train, y_test, "K-Nearest Neighbors")
score(rf_model, X_train_processed, X_test_processed, y_train, y_test, "Random Forest")
score(xgb_model, X_train_processed, X_test_processed, y_train, y_test, "XGBoost")

# Score ensemble models
score(stacking_model, X_train_processed, X_test_processed, y_train, y_test, "Stacking Model")
score(voting_model, X_train_processed, X_test_processed, y_train, y_test, "Voting Model")
"""

def score_zindi(model, base_pipeline, test_df, output_file='Zindi_submission_best_stacking.csv'):
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



##########################################################

# Execute for Zindi

# After training and selecting your best model
# best_model = grid_search.best_estimator_  # this might take too long again so re-think this!!!!!

# Load the test data for Zindi submission
zindi_test_data = pd.read_csv('../data/Test.csv')  # Save the Zindi test.csv into new name/ variable

# Create the Zindi submission
zindi_predictions, zindi_submission_df = score_zindi(best_stacking_model, base_pipeline, zindi_test_data) # this will create/ save new csv file in folder too

print(zindi_submission_df)