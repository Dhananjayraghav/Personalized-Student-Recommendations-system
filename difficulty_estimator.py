import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class DifficultyEstimator:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def train(self, interactions: pd.DataFrame, resource_features: pd.DataFrame):
        # Merge interaction data with resource features
        data = pd.merge(
            interactions,
            resource_features,
            left_on='resource_id',
            right_on='id'
        )

        # Calculate difficulty metric (time spent / correctness)
        data['difficulty'] = data['time_spent'] / (data['score'] + 0.01)  # Avoid division by zero

        # Features: resource characteristics
        X = data[['math_level', 'reading_level', 'resource_length', 'concept_complexity']]
        y = data['difficulty']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model trained with MAE: {mae:.2f}")
        self.is_trained = True

    def estimate_difficulty(self, resource_features: pd.DataFrame) -> float:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        return self.model.predict(resource_features)[0]

    def get_optimal_sequence(self, resources: List[Dict], student_skill: float) -> List[Dict]:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Estimate difficulty for all resources
        resource_df = pd.DataFrame(resources)
        difficulties = self.model.predict(resource_df[['math_level', 'reading_level',
                                                       'resource_length', 'concept_complexity']])

        # Calculate distance from student's skill level
        distances = np.abs(difficulties - student_skill)

        # Sort by closest to student's level
        sorted_indices = np.argsort(distances)

        return [resources[i] for i in sorted_indices]