import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeFilteringRecommender
from .knowledge_tracing import KnowledgeTracer
from semantic_analyzer import AdvancedNLPProcessor
from config.constants import RecommendationWeights


class HybridRecommender:
    def __init__(
            self,
            content_based_rec: ContentBasedRecommender,
            collab_rec: CollaborativeFilteringRecommender,
            knowledge_tracer: KnowledgeTracer,
            nlp_processor: AdvancedNLPProcessor,
            weights: Optional[Dict[str, float]] = None
    ):
        """
        Hybrid recommender that combines:
        - Content-based filtering
        - Collaborative filtering
        - Knowledge tracing
        - Social/NLP signals

        Args:
            weights: Dictionary of weights for each strategy:
                    {
                        'content': 0.4,
                        'collaborative': 0.3,
                        'knowledge': 0.2,
                        'social': 0.1
                    }
        """
        self.content_rec = content_based_rec
        self.collab_rec = collab_rec
        self.kt = knowledge_tracer
        self.nlp = nlp_processor
        self.weights = weights or RecommendationWeights.DEFAULT_HYBRID_WEIGHTS

    def recommend(self, student_id: int, n: int = 10, context: Optional[Dict] = None) -> List[Dict]:
        """Generate hybrid recommendations for a student"""
        # Get base recommendations from each strategy
        content_recs = self.content_rec.recommend(student_id, n * 3)
        collab_recs = self.collab_rec.recommend(student_id, n * 3)

        # Get knowledge state information
        skill_gaps = self.kt.get_skill_gaps(student_id)
        knowledge_recs = self._get_knowledge_based_recs(student_id, skill_gaps, n * 2)

        # Combine all recommendations
        combined_scores = self._combine_recommendations(
            content_recs,
            collab_recs,
            knowledge_recs
        )

        # Apply contextual filters if provided
        if context:
            combined_scores = self._apply_context(combined_scores, context)

        # Get top N recommendations
        sorted_recs = sorted(combined_scores.items(),
                             key=lambda x: x[1]['combined_score'],
                             reverse=True)

        return self._format_final_recommendations(sorted_recs[:n])

    def _get_knowledge_based_recs(self, student_id: int, skill_gaps: List[str], n: int) -> List[Dict]:
        """Get recommendations based on skill gaps"""
        if not skill_gaps:
            return []

        # Get resources targeting the skill gaps
        skill_resources = defaultdict(list)
        for skill in skill_gaps:
            # In a real system, you'd query resources tagged with this skill
            resources = self._query_resources_by_skill(skill)
            for res in resources:
                skill_resources[res['id']].append({
                    'skill': skill,
                    'relevance': res['skill_relevance']
                })

        # Format as recommendations
        return [{
            'resource_id': res_id,
            'type': 'knowledge',
            'score': sum(s['relevance'] for s in skills) / len(skills),
            'metadata': {
                'target_skills': [s['skill'] for s in skills],
                'explanation': f"Targets skill gaps: {', '.join([s['skill'] for s in skills])}"
            }
        } for res_id, skills in skill_resources.items()]

    def _combine_recommendations(self, *recommendation_lists: List[List[Dict]]) -> Dict:
        """Combine recommendations from different strategies"""
        combined = defaultdict(lambda: {
            'scores': defaultdict(float),
            'metadata': {},
            'combined_score': 0.0
        })

        # Aggregate scores from each recommendation type
        for rec_type, recs in zip(['content', 'collaborative', 'knowledge'], recommendation_lists):
            for rec in recs:
                res_id = rec['resource_id']
                combined[res_id]['scores'][rec_type] = rec['score']
                combined[res_id]['metadata'].update(rec.get('metadata', {}))

                # Add type-specific metadata
                if rec_type not in combined[res_id]['metadata']:
                    combined[res_id]['metadata'][rec_type] = True

        # Calculate weighted combined score
        for res_id in combined:
            total = 0.0
            for rec_type, weight in self.weights.items():
                if rec_type in combined[res_id]['scores']:
                    total += combined[res_id]['scores'][rec_type] * weight
            combined[res_id]['combined_score'] = total

        return combined

    def _apply_context(self, recommendations: Dict, context: Dict) -> Dict:
        """Apply contextual filters to recommendations"""
        filtered = {}

        for res_id, data in recommendations.items():
            # Apply time-based decay if resource is old
            if 'resource_age' in context:
                age = context['resource_age'].get(res_id, 0)
                decay = np.exp(-age / 365)  # Exponential decay over 1 year
                data['combined_score'] *= decay

            # Boost resources matching current session topic
            if 'session_topics' in context:
                topic_match = self._calculate_topic_match(res_id, context['session_topics'])
                data['combined_score'] *= (1.0 + topic_match)

            filtered[res_id] = data

        return filtered

    def _calculate_topic_match(self, resource_id: int, topics: List[str]) -> float:
        """Calculate how well resource matches current session topics"""
        # Get resource topics (from metadata or NLP analysis)
        resource_topics = self._get_resource_topics(resource_id)

        if not resource_topics or not topics:
            return 0.0

        # Calculate semantic similarity between topics
        topic_embeddings = self.nlp.get_text_embeddings(topics + resource_topics)
        session_embs = topic_embeddings[:len(topics)]
        resource_embs = topic_embeddings[len(topics):]

        # Max similarity between any pair of topics
        max_sim = max(
            self.nlp.calculate_similarity(se, re)
            for se in session_embs
            for re in resource_embs
        )

        return max_sim

    def _format_final_recommendations(self, recommendations: List) -> List[Dict]:
        """Format the final output with explanations"""
        return [{
            'resource_id': res_id,
            'score': data['combined_score'],
            'explanation': self._generate_explanation(data),
            'metadata': data['metadata']
        } for res_id, data in recommendations]

    def _generate_explanation(self, data: Dict) -> str:
        """Generate natural language explanation for the recommendation"""
        explanations = []

        if data['scores'].get('content', 0) > 0.7:
            explanations.append("matches your academic profile")

        if data['scores'].get('collaborative', 0) > 0.6:
            explanations.append("popular with similar students")

        if 'target_skills' in data['metadata']:
            skills = ", ".join(data['metadata']['target_skills'][:2])
            explanations.append(f"targets your skill gaps in {skills}")

        if 'session_topics' in data['metadata']:
            explanations.append("relevant to your current learning session")

        return "Recommended because " + ", ".join(explanations) if explanations else "Recommended for you"

    # Helper methods that would connect to your database
    def _query_resources_by_skill(self, skill: str) -> List[Dict]:
        """Mock method - in real system would query your database"""
        return [
            {'id': 101, 'skill_relevance': 0.9},
            {'id': 103, 'skill_relevance': 0.8}
        ]

    def _get_resource_topics(self, resource_id: int) -> List[str]:
        """Mock method - in real system would get from resource metadata"""
        return {
            101: ['mathematics', 'calculus'],
            102: ['literature', 'shakespeare'],
            103: ['programming', 'python'],
            104: ['history', 'ancient'],
            105: ['mathematics', 'linear-algebra']
        }.get(resource_id, [])