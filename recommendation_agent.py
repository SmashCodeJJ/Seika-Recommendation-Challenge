import os
from typing import List
import openai
from dotenv import load_dotenv
from data import Story, UserProfile
import random

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RecommendationAgent:
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
    
    def _calculate_story_score(self, story: Story, user_profile: UserProfile) -> float:
        """
        Calculate a score for how well a story matches a user profile
        """
        score = 0
        matched_preferences = 0
        matched_interests = 0
        matched_tags = 0
        has_anime_match = False
        exact_preference_matches = 0
        
        # Match with user's favorite anime (higher weight and partial matching)
        anime_score = 0
        for anime in user_profile.favorite_anime:
            # Direct title or tag match
            if anime.lower() in story.title.lower() or any(anime.lower() in tag.lower() for tag in story.tags):
                anime_score += 5.0  # Increased from 4.0
                has_anime_match = True
            
            # Partial matches for anime-related content
            anime_keywords = {
                'naruto': [
                    'ninja', 'shinobi', 'chakra', 'hokage', 'jutsu', 'dattebayo', 'rasengan', 'sharingan', 
                    'hidden leaf', 'ninja way', 'shadow clone', 'nine-tails', 'sage mode', 'ninja academy',
                    'chunin exam', 'anbu', 'rogue ninja', 'ninja tools', 'hand signs', 'ninja ranks'
                ],
                'demon slayer': [
                    'demon', 'slayer', 'sword', 'breathing technique', 'hashira', 'oni', 'total concentration',
                    'nichirin', 'corps', 'demon hunter', 'demon art', 'blood demon art', 'demon slayer mark',
                    'breath styles', 'demon slayer corps', 'final selection', 'demon moon', 'demon hunter exam',
                    'demon hunter training', 'demon hunter rank'
                ],
                'jujutsu kaisen': [
                    'sorcerer', 'curse', 'jujutsu', 'domain expansion', 'cursed energy', 'cursed technique',
                    'shaman', 'binding vow', 'cursed spirit', 'jujutsu high', 'grade curses', 'reverse curse',
                    'innate domain', 'cursed objects', 'jujutsu society', 'cursed womb', 'cursed tools',
                    'barrier technique', 'cursed restriction', 'jujutsu training'
                ]
            }
            
            # Add general anime tropes that match the user's interests
            shared_keywords = [
                'training arc', 'power up', 'tournament arc', 'friendship power', 'chosen one', 'mentor',
                'rival', 'hidden power', 'special technique', 'power system', 'training montage',
                'special move', 'ultimate form', 'power level', 'secret technique', 'ancient power',
                'forbidden technique', 'legendary weapon', 'special ability', 'power awakening'
            ]
            
            if anime.lower() in anime_keywords:
                for keyword in anime_keywords[anime.lower()]:
                    if (keyword in story.title.lower() or 
                        any(keyword in tag.lower() for tag in story.tags)):
                        anime_score += 2.0  # Increased from 1.5
                        has_anime_match = True
                
                # Check for shared anime tropes
                for keyword in shared_keywords:
                    if (keyword in story.title.lower() or 
                        any(keyword in tag.lower() for tag in story.tags)):
                        anime_score += 1.5  # Increased from 1.0
                        has_anime_match = True
        
        score += anime_score
        
        # Match with user's preferred tags (using weights)
        for tag in user_profile.preferred_tags:
            if tag.lower() in [t.lower() for t in story.tags]:
                weight = self.base_tag_weights.get(tag.lower(), 1.0)
                score += weight
                matched_tags += 1
        
        # Match with user's interests
        for interest in user_profile.interests:
            if interest.lower() in story.title.lower() or any(interest.lower() in tag.lower() for tag in story.tags):
                score += 2.0  # Increased from 1.5
                matched_interests += 1
        
        # Match with user's preferences
        for preference in user_profile.preferences:
            if preference.lower() in story.title.lower() or any(preference.lower() in tag.lower() for tag in story.tags):
                score += 2.5  # Increased from 1.8
                matched_preferences += 1
                # Check for exact matches
                if preference.lower() in [t.lower() for t in story.tags]:
                    exact_preference_matches += 1
        
        # Apply multiplier for multiple matches
        multiplier = 1.0
        if matched_preferences >= 2:
            multiplier += 0.2  # 20% bonus for matching multiple preferences
        if matched_interests >= 2:
            multiplier += 0.1  # 10% bonus for matching multiple interests
        if matched_tags >= 2:
            multiplier += 0.1  # 10% bonus for matching multiple tags
            
        # Special bonus for matching both anime and themes
        if has_anime_match and (matched_preferences > 0 or matched_interests > 0 or matched_tags > 0):
            multiplier += 0.3  # 30% bonus for matching both anime and themes
            
        # Extra bonus for exact preference matches
        if exact_preference_matches > 0:
            multiplier += 0.2 * exact_preference_matches  # 20% bonus per exact match
            
        return score * multiplier
    
    def get_recommendations(self, user_profile: UserProfile, prompt: str = None, num_recommendations: int = 10) -> List[str]:
        """
        Get story recommendations for a user profile
        """
        # Calculate scores for all stories
        story_scores = [(story.id, self._calculate_story_score(story, user_profile)) for story in self.stories]
        
        # Sort by score and get top recommendations
        story_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [story_id for story_id, _ in story_scores[:num_recommendations]]
        
        # Ensure no duplicates
        return list(dict.fromkeys(recommendations)) 