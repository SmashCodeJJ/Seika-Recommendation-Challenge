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
        
    def get_recommendations(self, user_profile: UserProfile, prompt: str, num_recommendations: int = 10) -> List[str]:
        """
        Get story recommendations based on user profile and optimized prompt
        """
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
        
        full_prompt = f"""
        {prompt}
        
        Stories:
        {stories_str}
        
        User Profile:
        {user_profile_str}
        
        Please recommend {num_recommendations} stories that would be most relevant to this user.
        Return only the story IDs in a comma-separated list, ordered by relevance (most relevant first).
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a story recommendation system that matches stories to user preferences."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            # Parse and validate recommended IDs
            recommended_ids = response.choices[0].message.content.strip().split(',')
            recommended_ids = [id.strip() for id in recommended_ids]
            
            # Filter out invalid IDs and ensure we have the right number
            valid_ids = [id for id in recommended_ids if any(s.id == id for s in self.stories)]
            
            # If we don't have enough valid recommendations, add some random ones
            while len(valid_ids) < num_recommendations:
                remaining_ids = [s.id for s in self.stories if s.id not in valid_ids]
                if not remaining_ids:
                    break
                valid_ids.append(random.choice(remaining_ids))
            
            return valid_ids[:num_recommendations]
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            # Return random recommendations as fallback
            return [s.id for s in random.sample(self.stories, min(num_recommendations, len(self.stories)))] 