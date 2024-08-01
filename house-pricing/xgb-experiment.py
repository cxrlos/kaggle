import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import json
import sys


def load_config(experiment_id):
    with open('./configs/xgb-experiment.json', 'r') as f:
        configs = json.load(f)
    common_config = configs['common']
    experiment_config = configs[experiment_id]
    return {**common_config, **experiment_config}


def preprocess_data(config, train_data, test_data):
    X = train_data.drop(['Id', 'SalePrice'], axis=1)
    y = train_data['SalePrice']
    test_features = test_data.drop('Id', axis=1)

    all_features = pd.concat([X, test_features])

    numeric_features = all_features.select_dtypes(include=[np.number]).columns
    categorical_features = all_features.select_dtypes(exclude=[np.number]).columns

    numeric_imputer = SimpleImputer(strategy=config['preprocessing']['numeric_imputer_strategy'])
    all_features[numeric_features] = numeric_imputer.fit_transform(all_features[numeric_features])

    categorical_imputer = SimpleImputer(strategy=config['preprocessing']['categorical_imputer_strategy'])
    all_features[categorical_features] = categorical_imputer.fit_transform(all_features[categorical_features])

    label_encoder = LabelEncoder()
    for col in categorical_features:
        all_features[col] = label_encoder.fit_transform(all_features[col].astype(str))

    X = all_features.iloc[:len(X), :]
    test_features = all_features.iloc[len(X):, :]

    return X, y, test_features


def train_model(config, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config['training']['test_size'], random_state=config['model']['random_state'])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        config['model'],
        dtrain,
        num_boost_round=config['model']['n_estimators'],
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=config['training']['early_stopping_rounds'],
        verbose_eval=config['training']['verbose_eval']
    )

    val_preds = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse}")

    return model


def main(experiment_id):
    common_config = load_config('common')
    config = load_config(experiment_id)
    print(f"Common config: {common_config}")

    train_data_dir = f"data/{common_config['data']['train_file']}"
    test_data_dir = f"data/{common_config['data']['test_file']}"
    train_data = pd.read_csv(train_data_dir)
    test_data = pd.read_csv(test_data_dir)

    X, y, test_features = preprocess_data(config, train_data, test_data)

    model = train_model(config, X, y)

    dtest = xgb.DMatrix(test_features)
    test_preds = model.predict(dtest)

    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': test_preds
    })
    submission.to_csv(f'submissions/xgb-experiment/{experiment_id}.csv', index=False)

    print(f"Submission file for experiment {experiment_id} created successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <experiment_id>")
        sys.exit(1)
    experiment_id = sys.argv[1]
    main(experiment_id)
