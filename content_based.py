import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from typing import List, Dict


class ContentBasedRecommender:
    def __init__(self, students_df: pd.DataFrame, resources_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.students = students_df
        self.resources = resources_df
        self.interactions = interactions_df
        self.scaler = MinMaxScaler()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare student and resource features"""
        # Student features
        student_features = ['math_score', 'reading_score', 'grade_level', 'learning_style']
        self.student_features = self.scaler.fit_transform(
            pd.get_dummies(self.students[student_features], columns=['learning_style'])
        )

        # Resource features
        resource_features = ['math_level', 'reading_level', 'grade_level', 'resource_type']
        self.resource_features = self.scaler.transform(
            pd.get_dummies(self.resources[resource_features], columns=['resource_type'])
        )

        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(
            self.student_features,
            self.resource_features
        )

    def recommend(self, student_id: int, n: int = 5) -> List[Dict]:
        """Generate content-based recommendations"""
        if student_id not in self.students['id'].values:
            return []

        student_idx = self.students[self.students['id'] == student_id].index[0]
        scores = self.similarity_matrix[student_idx]

        # Apply time decay
        time_decay = np.array([
            0.9 if (datetime.now() - pd.to_datetime(r)).days < 30 else
            0.7 if (datetime.now() - pd.to_datetime(r)).days < 90 else
            0.5 for r in self.resources['publish_date']
        ])

        # Combine with popularity
        final_scores = scores * time_decay * self.resources['popularity']

        # Get top N recommendations
        top_indices = np.argsort(final_scores)[-n:][::-1]
        return [{
            'resource_id': self.resources.iloc[idx]['id'],
            'title': self.resources.iloc[idx]['title'],
            'score': float(final_scores[idx]),
            'type': 'content',
            'metadata': {
                'match_features': self._get_matching_features(student_idx, idx),
                'content_type': self.resources.iloc[idx]['resource_type']
            }
        } for idx in top_indices]

    def _get_matching_features(self, student_idx: int, resource_idx: int) -> List[str]:
        """Identify which features contributed most to the match"""
        student_vec = self.student_features[student_idx]
        resource_vec = self.resource_features[resource_idx]
        feature_names = pd.get_dummies(
            self.students[['math_score', 'reading_score', 'grade_level', 'learning_style']],
            columns=['learning_style']
        ).columns

        top_features = []
        for i in range(len(feature_names)):
            if student_vec[i] > 0.7 and resource_vec[i] > 0.7:
                top_features.append(feature_names[i])

        return top_features[:3]