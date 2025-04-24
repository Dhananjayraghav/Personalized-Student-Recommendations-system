# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(title="Advanced Student Recommendation System")

# ---------- Data Models ----------
class Student(BaseModel):
    id: int
    math_score: float
    reading_score: float
    grade_level: int
    learning_style: str

class Resource(BaseModel):
    id: int
    title: str
    math_level: float
    reading_level: float
    tags: List[str]
    resource_type: str

class Interaction(BaseModel):
    student_id: int
    resource_id: int
    rating: float
    time_spent: float

# ---------- Core Components ----------
class ContentBasedRecommender:
    def __init__(self, students: List[Student], resources: List[Resource]):
        self.students = students
        self.resources = resources
        self.scaler = MinMaxScaler()
        self._prepare_features()

    def _prepare_features(self):
        # Implementation from previous content-based recommender
        pass

    def recommend(self, student_id: int, n: int = 5) -> List[Dict]:
        # Implementation from previous content-based recommender
        pass

class CollaborativeFilteringRecommender:
    def __init__(self, interactions: List[Interaction]):
        self.interactions = interactions
        self._prepare_rating_matrix()

    def _prepare_rating_matrix(self):
        # Implementation from previous collaborative filtering
        pass

    def recommend(self, student_id: int, n: int = 5) -> List[Dict]:
        # Implementation from previous collaborative filtering
        pass

class HybridRecommender:
    def __init__(self, content_rec: ContentBasedRecommender, 
                 collab_rec: CollaborativeFilteringRecommender):
        self.content_rec = content_rec
        self.collab_rec = collab_rec

    def recommend(self, student_id: int, n: int = 5) -> List[Dict]:
        # Combine recommendations from both strategies
        content_recs = self.content_rec.recommend(student_id, n*2)
        collab_recs = self.collab_rec.recommend(student_id, n*2)
        
        # Combine and deduplicate
        all_recs = {r['resource_id']: r for r in content_recs + collab_recs}
        
        # Sort by combined score
        sorted_recs = sorted(all_recs.values(), 
                           key=lambda x: x['score'], 
                           reverse=True)
        return sorted_recs[:n]

# ---------- API Setup ----------
students_db = [
    Student(id=1, math_score=90, reading_score=85, grade_level=10, learning_style="visual"),
    Student(id=2, math_score=70, reading_score=95, grade_level=9, learning_style="verbal")
]

resources_db = [
    Resource(id=101, title="Advanced Calculus", math_level=90, reading_level=60, 
            tags=["math", "calculus"], resource_type="video"),
    Resource(id=102, title="Literature Analysis", math_level=40, reading_level=95, 
            tags=["literature"], resource_type="article")
]

interactions_db = []

@app.on_event("startup")
async def startup_event():
    # Initialize recommenders
    global hybrid_recommender
    content_rec = ContentBasedRecommender(students_db, resources_db)
    collab_rec = CollaborativeFilteringRecommender(interactions_db)
    hybrid_recommender = HybridRecommender(content_rec, collab_rec)

# ---------- API Endpoints ----------
@app.get("/recommend/{student_id}", response_model=List[Dict])
async def get_recommendations(student_id: int, n: int = 5):
    """Get hybrid recommendations for a student"""
    try:
        return hybrid_recommender.recommend(student_id, n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def record_feedback(interaction: Interaction):
    """Record student interaction with recommended resource"""
    try:
        interactions_db.append(interaction.dict())
        # Reinitialize recommenders with new data
        startup_event()
        return {"message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/content/{student_id}", response_model=List[Dict])
async def get_content_recommendations(student_id: int, n: int = 5):
    """Get content-based recommendations"""
    try:
        return hybrid_recommender.content_rec.recommend(student_id, n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/collaborative/{student_id}", response_model=List[Dict])
async def get_collaborative_recommendations(student_id: int, n: int = 5):
    """Get collaborative filtering recommendations"""
    try:
        return hybrid_recommender.collab_rec.recommend(student_id, n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
