import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
import json
import sys


def load_config(experiment_id):
    with open('./configs/dnn_experiment.json', 'r') as f:
        configs = json.load(f)
    common_config = configs['common']
    experiment_config = configs[experiment_id]
    return {**common_config, **experiment_config}


def preprocess_data(config, train_data, test_data):
    # Separate features and target
    X = train_data.drop(['Id', 'SalePrice'], axis=1)
    y = train_data['SalePrice']
    test_features = test_data.drop('Id', axis=1)

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config['preprocessing']['numeric_imputer_strategy'])),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config['preprocessing']['categorical_imputer_strategy'])),
        ('encoder', OneHotEncoder())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    test_features_processed = preprocessor.transform(test_features)

    return X_processed, y, test_features_processed


def create_model(config, input_shape):
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Input(shape=input_shape))
    
    # Hidden layers
    for units in config['model']['hidden_layers']:
        model.add(keras.layers.Dense(units, activation=config['model']['activation']))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(config['model']['dropout_rate']))
    
    # Output layer
    model.add(keras.layers.Dense(1))
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['model']['learning_rate']),
                  loss='mean_squared_error')
    
    return model


def train_model(config, X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config['training']['test_size'],
        random_state=config['model']['random_state']
    )

    model = create_model(config, input_shape=(X_train.shape[1],))

    # Prepare callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=config['training']['reduce_lr_factor'],
        patience=config['training']['reduce_lr_patience']
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate the model
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse}")

    return model, history


def main(experiment_id):
    config = load_config(experiment_id)
    print(f"Experiment config: {config}")

    train_data_dir = f"data/{config['data']['train_file']}"
    test_data_dir = f"data/{config['data']['test_file']}"
    train_data = pd.read_csv(train_data_dir)
    test_data = pd.read_csv(test_data_dir)

    X, y, test_features = preprocess_data(config, train_data, test_data)

    model, history = train_model(config, X, y)

    # Make predictions on test data
    test_preds = model.predict(test_features).flatten()

    # Create submission file
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': test_preds
    })
    submission.to_csv(f'submissions/dnn_experiment/dnn-{experiment_id}.csv', index=False)

    print(f"Submission file for experiment {experiment_id} created successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dnn_experiment.py <experiment_id>")
        sys.exit(1)
    experiment_id = sys.argv[1]
    main(experiment_id)
