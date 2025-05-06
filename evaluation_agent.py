import os
from typing import List, Tuple, Dict
import openai
from dotenv import load_dotenv
from data import Story, UserProfile
import numpy as np
from collections import Counter
import re

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
        Get ground truth recommendations using GPT-3.5-turbo
        """
        # Prepare the stories for evaluation
        stories_str = "\n".join([
            f"ID: {story.id}\nTitle: {story.title}\nTags: {', '.join(story.tags)}\n"
            for story in self.stories
        ])
        
        user_profile_str = f"""
        User Preferences: {user_profile.preferences}
        Interests: {', '.join(user_profile.interests)}
        Favorite Anime: {', '.join(user_profile.favorite_anime)}
        Preferred Tags: {', '.join(user_profile.preferred_tags)}
        """
        
        prompt = f"""
        You are an expert story recommendation evaluator. Your task is to select the most relevant stories for a user based on their profile.
        
        User Profile:
        {user_profile_str}
        
        Available Stories:
        {stories_str}
        
        Please select the top {num_recommendations} most relevant stories for this user based on:
        1. How well they match the user's preferences and interests
        2. How well they align with the user's favorite anime
        3. How well they cover the user's preferred tags
        4. The diversity and relevance of the recommendations
        
        Return ONLY the story IDs in order of relevance, separated by commas. Do not include any other text or formatting.
        Example format: 123456, 234567, 345678
        """
        
        try:
            print(f"\nSending request to GPT with {len(self.stories)} stories...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert story recommendation evaluator. Return ONLY the story IDs in a comma-separated list."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            print(f"\nGPT Response: {content}")
            
            # Extract IDs using regex to handle various formats
            recommended_ids = re.findall(r'\d+', content)
            print(f"\nParsed IDs: {recommended_ids}")
            
            # Filter IDs to only those that exist in our stories
            valid_ids = [id for id in recommended_ids if any(s.id == id for s in self.stories)]
            print(f"\nValid IDs (found in stories): {valid_ids}")
            
            # Print the ground truth recommendations
            print("\nGround Truth Recommendations:")
            print("-" * 50)
            for story_id in valid_ids[:10]:
                story = next(s for s in self.stories if s.id == story_id)
                print(f"{story.title} (ID: {story_id})")
            
            return valid_ids[:num_recommendations]
            
        except Exception as e:
            print(f"Error in GPT ground truth generation: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"API Response: {e.response}")
            return []
    
    def evaluate_recommendations(self, recommended_ids: List[str], ground_truth_ids: List[str], user_profile: UserProfile) -> Tuple[float, List[str]]:
        """
        Evaluate recommendations using GPT-3.5-turbo
        Returns: (score, feedback)
        """
        # Get the stories for evaluation
        recommended_stories = [s for s in self.stories if s.id in recommended_ids]
        ground_truth_stories = [s for s in self.stories if s.id in ground_truth_ids]
        
        # Prepare the evaluation prompt
        stories_str = "\n".join([
            f"ID: {story.id}\nTitle: {story.title}\nTags: {', '.join(story.tags)}\n"
            for story in recommended_stories
        ])
        
        ground_truth_str = "\n".join([
            f"ID: {story.id}\nTitle: {story.title}\nTags: {', '.join(story.tags)}\n"
            for story in ground_truth_stories
        ])
        
        user_profile_str = f"""
        User Preferences: {user_profile.preferences}
        Interests: {', '.join(user_profile.interests)}
        Favorite Anime: {', '.join(user_profile.favorite_anime)}
        Preferred Tags: {', '.join(user_profile.preferred_tags)}
        """
        
        prompt = f"""
        You are an expert story recommendation evaluator. Your task is to evaluate how well the recommended stories match the user's profile and preferences.
        
        User Profile:
        {user_profile_str}
        
        Recommended Stories:
        {stories_str}
        
        Ground Truth Stories (what the user should ideally like):
        {ground_truth_str}
        
        Please evaluate the recommendations based on:
        1. How well they match the user's preferences and interests
        2. How well they align with the user's favorite anime
        3. How well they cover the user's preferred tags
        4. The diversity and relevance of the recommendations
        
        Provide:
        1. A score between 0 and 1 (1 being perfect match)
        2. Detailed feedback on what worked well and what could be improved
        3. Specific suggestions for better recommendations
        
        Format your response as:
        Score: [number between 0 and 1]
        Feedback:
        - [detailed feedback point 1]
        - [detailed feedback point 2]
        ...
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert story recommendation evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse the response
            content = response.choices[0].message.content
            score_line = [line for line in content.split('\n') if line.startswith('Score:')][0]
            score = float(score_line.split(':')[1].strip())
            
            feedback_lines = [line.strip('- ') for line in content.split('\n') 
                            if line.startswith('-')]
            
            return score, feedback_lines
            
        except Exception as e:
            print(f"Error in GPT evaluation: {str(e)}")
            return 0.0, ["Error in evaluation. Please try again."] 