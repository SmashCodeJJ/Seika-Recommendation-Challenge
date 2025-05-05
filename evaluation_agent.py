import os
from typing import List, Tuple, Dict
import openai
from dotenv import load_dotenv
from data import Story, UserProfile
import numpy as np

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class EvaluationAgent:
    def __init__(self, stories: List[Story]):
        self.stories = stories
        self.tag_weights = {
            'power-fantasy': 1.5,
            'isekai': 1.4,
            'crossover': 1.3,
            'underdog': 1.2,
            'romance': 1.1,
            'action': 1.0,
            'comedy': 0.9,
            'drama': 0.9,
            'supernatural': 1.2,
            'sci-fi': 1.0
        }
        
    def get_ground_truth_recommendations(self, user_profile: UserProfile, num_recommendations: int = 10) -> List[str]:
        """
        Get ground truth recommendations directly from the full user profile
        """
        # Calculate story scores based on user preferences
        story_scores = []
        for story in self.stories:
            score = 0
            
            # Match with user's favorite anime
            for anime in user_profile.favorite_anime:
                if anime.lower() in story.title.lower() or any(anime.lower() in tag.lower() for tag in story.tags):
                    score += 2.0
            
            # Match with user's preferred tags
            for tag in user_profile.preferred_tags:
                if tag.lower() in [t.lower() for t in story.tags]:
                    score += self.tag_weights.get(tag.lower(), 1.0)
            
            # Match with user's interests
            for interest in user_profile.interests:
                if interest.lower() in story.title.lower() or any(interest.lower() in tag.lower() for tag in story.tags):
                    score += 1.5
            
            story_scores.append((story.id, score))
        
        # Sort by score and get top recommendations
        story_scores.sort(key=lambda x: x[1], reverse=True)
        return [story_id for story_id, _ in story_scores[:num_recommendations]]
    
    def evaluate_recommendations(self, recommended_ids: List[str], ground_truth_ids: List[str], user_profile: UserProfile = None) -> Tuple[float, List[str]]:
        """
        Evaluate recommendations against ground truth using multiple metrics
        Returns: (score, feedback)
        """
        # Clean and normalize IDs
        recommended_ids = [id.strip() for id in recommended_ids if id.strip()]
        ground_truth_ids = [id.strip() for id in ground_truth_ids if id.strip()]
        
        # Calculate precision@10
        correct_recommendations = set(recommended_ids) & set(ground_truth_ids)
        precision = len(correct_recommendations) / len(recommended_ids) if recommended_ids else 0
        
        # Calculate recall
        recall = len(correct_recommendations) / len(ground_truth_ids) if ground_truth_ids else 0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Generate detailed feedback
        feedback = []
        
        if precision < 0.5:
            feedback.append("The recommendations have low precision. Consider better matching user preferences.")
        if recall < 0.5:
            feedback.append("The recommendations are missing many relevant stories. Consider broader matching criteria.")
        
        # Check for diversity
        unique_recommendations = len(set(recommended_ids))
        if unique_recommendations < len(recommended_ids):
            feedback.append("There are duplicate recommendations. Ensure each story is recommended only once.")
        
        if user_profile:
            # Check for relevance to user's favorite anime
            recommended_stories = [s for s in self.stories if s.id in recommended_ids]
            anime_matches = sum(1 for story in recommended_stories 
                              if any(anime.lower() in story.title.lower() or 
                                    any(anime.lower() in tag.lower() for tag in story.tags)
                                    for anime in user_profile.favorite_anime))
            if anime_matches < 3:
                feedback.append("Consider including more stories related to the user's favorite anime.")
            
            # Check for tag coverage
            user_tags = set(tag.lower() for tag in user_profile.preferred_tags)
            story_tags = set(tag.lower() for story in recommended_stories for tag in story.tags)
            tag_coverage = len(user_tags & story_tags) / len(user_tags) if user_tags else 0
            if tag_coverage < 0.5:
                feedback.append("The recommendations could better cover the user's preferred tags.")
        
        return f1_score, feedback 