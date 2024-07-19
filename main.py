from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

if __name__ == '__main__':
    # Download dataset
    from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
    download_robot_execution_failures()

    pipeline = Pipeline([
                ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
                ('classifier', RandomForestClassifier()),
                ])

    df_ts, y = load_robot_execution_failures()
    X = pd.DataFrame(index=y.index)

    pipeline.set_params(augmenter__timeseries_container=df_ts)
    pipeline.fit(X, y)

    y_pred = pipeline.predict(X)

    print("Predictions:", y_pred)

    accuracy = accuracy_score(y, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y, y_pred))