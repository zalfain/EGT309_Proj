import pandas as pd


class DataPreprocessor:
    """Clean machine sensor data and engineer useful features."""

    def __init__(self, config):
        self.drop_columns = config.get("drop_columns", [])
        self.imputation_strategy = config.get("imputation_strategy", "median")
        self.negative_vibration_as_missing = config.get(
            "negative_vibration_as_missing", True
        )
        self.timestamp_features = config.get(
            "timestamp_features", ["hour", "day_of_week", "month"]
        )

        self.medians = {}

    def fit(self, df):
        """Calculate imputation values using training data."""

        data = df.copy()

        if self.negative_vibration_as_missing and "vibration" in data.columns:
            data.loc[data["vibration"] < 0, "vibration"] = float("nan")

        if self.imputation_strategy != "median":
            raise ValueError("Only median imputation is currently supported.")

        for column in ["temperature", "vibration"]:
            if column in data.columns:
                median = data[column].median()

                if pd.isna(median):
                    raise ValueError(f"Cannot calculate median for {column}")

                self.medians[column] = median

        return self

    def transform(self, df):
        """Clean data and generate timestamp features."""

        if not self.medians:
            raise ValueError("Preprocessor must be fitted first.")

        data = df.copy()

        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])

            feature_extractors = {
                "hour": data["timestamp"].dt.hour,
                "day_of_week": data["timestamp"].dt.dayofweek,
                "month": data["timestamp"].dt.month
            }

            for feature in self.timestamp_features:
                if feature in feature_extractors:
                    data[feature] = feature_extractors[feature]

            data = data.drop(columns=["timestamp"])

        if self.negative_vibration_as_missing and "vibration" in data.columns:
            data.loc[data["vibration"] < 0, "vibration"] = float("nan")

        for column, median in self.medians.items():
            data[column] = data[column].fillna(median)

        data = data.drop(columns=self.drop_columns, errors="ignore")

        return data

    def fit_transform(self, df):
        """Fit and transform the supplied dataset."""

        return self.fit(df).transform(df)