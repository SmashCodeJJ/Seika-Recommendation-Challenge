import os
from typing import List, Tuple, Dict
import openai
from dotenv import load_dotenv
from data import Story, UserProfile
import numpy as np
from collections import Counter

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class EvaluationAgent:
    def __init__(self, stories: List[Story]):
        self.stories = stories
        self.base_tag_weights = {
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
        
    def _generate_dynamic_weights(self, user_profile: UserProfile) -> Dict[str, float]:
        """
        Generate dynamic tag weights based on user profile
        """
        # Start with base weights
        weights = self.base_tag_weights.copy()
        
        # Analyze user preferences and interests
        all_user_tags = (
            user_profile.preferred_tags + 
            user_profile.interests + 
            [tag.lower() for anime in user_profile.favorite_anime for tag in anime.split('-')]
        )
        
        # Count frequency of each tag
        tag_counts = Counter(all_user_tags)
        
        # Update weights based on frequency
        max_count = max(tag_counts.values()) if tag_counts else 1
        for tag, count in tag_counts.items():
            # Normalize count to weight between 1.0 and 2.0
            weight = 1.0 + (count / max_count)
            weights[tag] = max(weights.get(tag, 1.0), weight)
        
        # Special handling for anime-specific tags
        for anime in user_profile.favorite_anime:
            anime_tags = anime.lower().split('-')
            for tag in anime_tags:
                weights[tag] = max(weights.get(tag, 1.0), 1.5)  # Boost anime-related tags
        
        return weights
    
    def _calculate_tag_combination_score(self, story_tags: List[str], user_tags: List[str], weights: Dict[str, float]) -> float:
        """
        Calculate score based on tag combinations and their relationships
        """
        score = 0.0
        story_tag_set = set(tag.lower() for tag in story_tags)
        user_tag_set = set(tag.lower() for tag in user_tags)
        
        # Base score for individual tag matches
        for tag in user_tag_set:
            if tag in story_tag_set:
                score += weights.get(tag, 1.0)
        
        # Bonus for matching multiple related tags
        related_tag_groups = [
            {'underdog', 'reluctant hero', 'trauma healing'},
            {'found family', 'teamwork', 'loyalty'},
            {'supernatural', 'magic', 'fantasy'},
            {'romance', 'forbidden love', 'one-on-one romance'},
            {'power fantasy', 'epic battles', 'tournament arc'}
        ]
        
        for group in related_tag_groups:
            matches = len(group & story_tag_set & user_tag_set)
            if matches >= 2:  # Bonus for matching at least 2 tags from a related group
                score += matches * 0.5  # Add bonus points for each matching tag in the group
        
        return score
    
    def get_ground_truth_recommendations(self, user_profile: UserProfile, num_recommendations: int = 10) -> List[str]:
        """
        Get ground truth recommendations directly from the full user profile
        """
        # Get dynamic weights for this user
        tag_weights = self._generate_dynamic_weights(user_profile)
        
        # Calculate story scores based on user preferences
        story_scores = []
        print("\nDetailed Scoring Analysis:")
        print("-" * 50)
        
        for story in self.stories:
            score = 0
            score_breakdown = []
            
            # Match with user's favorite anime
            anime_score = 0
            for anime in user_profile.favorite_anime:
                if anime.lower() in story.title.lower() or any(anime.lower() in tag.lower() for tag in story.tags):
                    anime_score += 2.0
            if anime_score > 0:
                score += anime_score
                score_breakdown.append(f"Anime matches: +{anime_score}")
            
            # Match with user's preferred tags
            tag_score = 0
            matched_tags = []
            for tag in user_profile.preferred_tags:
                if tag.lower() in [t.lower() for t in story.tags]:
                    weight = tag_weights.get(tag.lower(), 1.0)
                    tag_score += weight
                    matched_tags.append(tag)
            if tag_score > 0:
                score += tag_score
                score_breakdown.append(f"Tag matches ({', '.join(matched_tags)}): +{tag_score:.1f}")
            
            # Match with user's interests
            interest_score = 0
            matched_interests = []
            for interest in user_profile.interests:
                if interest.lower() in story.title.lower() or any(interest.lower() in tag.lower() for tag in story.tags):
                    interest_score += 1.5
                    matched_interests.append(interest)
            if interest_score > 0:
                score += interest_score
                score_breakdown.append(f"Interest matches ({', '.join(matched_interests)}): +{interest_score:.1f}")
            
            # Additional scoring for preferences
            pref_score = 0
            matched_prefs = []
            for preference in user_profile.preferences:
                if preference.lower() in story.title.lower() or any(preference.lower() in tag.lower() for tag in story.tags):
                    pref_score += 1.8
                    matched_prefs.append(preference)
            if pref_score > 0:
                score += pref_score
                score_breakdown.append(f"Preference matches ({', '.join(matched_prefs)}): +{pref_score:.1f}")
            
            story_scores.append((story.id, score))
            
            if score > 0:
                print(f"\nStory: {story.title} (ID: {story.id})")
                print(f"Tags: {', '.join(story.tags)}")
                print(f"Total Score: {score:.1f}")
                for breakdown in score_breakdown:
                    print(f"  {breakdown}")
        
        # Sort by score and get top recommendations
        story_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Ground Truth Recommendations:")
        print("-" * 50)
        for story_id, score in story_scores[:10]:
            story = next(s for s in self.stories if s.id == story_id)
            print(f"{story.title} (ID: {story_id}) - Score: {score:.1f}")
        
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
            # Get dynamic weights for this user
            tag_weights = self._generate_dynamic_weights(user_profile)
            
            # Check for relevance to user's favorite anime
            recommended_stories = [s for s in self.stories if s.id in recommended_ids]
            anime_matches = sum(1 for story in recommended_stories 
                              if any(anime.lower() in story.title.lower() or 
                                    any(anime.lower() in tag.lower() for tag in story.tags)
                                    for anime in user_profile.favorite_anime))
            if anime_matches < 3:
                feedback.append("Consider including more stories related to the user's favorite anime.")
            
            # Check for tag coverage with dynamic weights
            user_tags = set(tag.lower() for tag in user_profile.preferred_tags)
            story_tags = set(tag.lower() for story in recommended_stories for tag in story.tags)
            tag_coverage = len(user_tags & story_tags) / len(user_tags) if user_tags else 0
            if tag_coverage < 0.5:
                feedback.append("The recommendations could better cover the user's preferred tags.")
            
            # Check for preference coverage
            user_preferences = set(pref.lower() for pref in user_profile.preferences)
            story_preferences = set(pref.lower() for story in recommended_stories 
                                  for pref in user_profile.preferences 
                                  if pref.lower() in story.title.lower() or 
                                  any(pref.lower() in tag.lower() for tag in story.tags))
            pref_coverage = len(story_preferences) / len(user_preferences) if user_preferences else 0
            if pref_coverage < 0.5:
                feedback.append("The recommendations could better match the user's preferences.")
        
        return f1_score, feedback 