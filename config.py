from enum import Enum


class RecommendationWeights(Enum):
    DEFAULT_HYBRID_WEIGHTS = {
        'content': 0.5,
        'collaborative': 0.3,
        'knowledge': 0.2
    }

    CONTENT_ONLY_WEIGHTS = {
        'content': 1.0,
        'collaborative': 0.0,
        'knowledge': 0.0
    }

    COLLAB_ONLY_WEIGHTS = {
        'content': 0.0,
        'collaborative': 1.0,
        'knowledge': 0.0
    }


class SkillLevels(Enum):
    BEGINNER = 0.3
    INTERMEDIATE = 0.6
    ADVANCED = 0.9


class ResourceTypes(Enum):
    VIDEO = 'video'
    ARTICLE = 'article'
    EXERCISE = 'exercise'
    QUIZ = 'quiz'