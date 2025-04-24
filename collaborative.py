import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


class CollaborativeFilteringRecommender:
    def __init__(self, students_df: pd.DataFrame, resources_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.students = students_df
        self.resources = resources_df
        self.interactions = interactions_df
        self._prepare_rating_matrix()

    def _prepare_rating_matrix(self):
        """Create user-item rating matrix"""
        # Create rating matrix (students x resources)
        self.rating_matrix = self.interactions.pivot_table(
            index='student_id',
            columns='resource_id',
            values='rating',
            fill_value=0
        )

        # Add missing students/resources
        all_students = set(self.students['id'])
        all_resources = set(self.resources['id'])
        self.rating_matrix = self.rating_matrix.reindex(
            index=all_students,
            columns=all_resources,
            fill_value=0
        )

        # Calculate user-user similarity
        self.user_similarity = cosine_similarity(self.rating_matrix)

    def recommend(self, student_id: int, n: int = 5) -> List[Dict]:
        """Generate collaborative recommendations"""
        if student_id not in self.rating_matrix.index:
            return []

        student_idx = self.rating_matrix.index.get_loc(student_id)
        user_ratings = self.rating_matrix.iloc[student_idx]

        # Find similar users
        similar_users = np.argsort(-self.user_similarity[student_idx])

        # Predict ratings for unrated items
        predicted_ratings = {}
        for resource_id in self.rating_matrix.columns:
            if user_ratings[resource_id] == 0:  # Only predict for unrated items
                weighted_sum = 0
                sim_sum = 0

                for other_idx in similar_users[1:6]:  # Top 5 similar users
                    other_rating = self.rating_matrix.iloc[other_idx][resource_id]
                    if other_rating > 0:
                        similarity = self.user_similarity[student_idx][other_idx]
                        weighted_sum += similarity * other_rating
                        sim_sum += similarity

                if sim_sum > 0:
                    predicted_ratings[resource_id] = weighted_sum / sim_sum

        # Get top N recommendations
        top_resources = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{
            'resource_id': res_id,
            'score': float(score),
            'type': 'collaborative',
            'metadata': {
                'similar_users_rated': self._get_similar_users_rated(student_idx, res_id),
                'predicted_rating': float(score)
            }
        } for res_id, score in top_resources]

    def _get_similar_users_rated(self, student_idx: int, resource_id: int) -> int:
        """Count how many similar users rated this resource"""
        similar_users = np.argsort(-self.user_similarity[student_idx])[1:6]  # Top 5 similar users
        return sum(
            1 for other_idx in similar_users
            if self.rating_matrix.iloc[other_idx][resource_id] > 0
        )