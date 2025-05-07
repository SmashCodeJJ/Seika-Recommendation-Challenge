import os
from typing import List
import openai
from dotenv import load_dotenv
from data import Story, UserProfile
import random
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RecommendationAgent:
    def __init__(self, stories: List[Story]):
        self.stories = stories
        # Enhanced base weights for different types of tags
        self.base_tag_weights = {
            # Power and Fantasy
            'power-fantasy': 2.5,  # Increased from 2.0
            'isekai': 2.2,        # Increased from 1.8
            'supernatural': 2.0,   # Increased from 1.5
            'magic': 1.8,         # Increased from 1.5
            
            # Character Dynamics
            'reluctant-hero': 2.3,  # New high priority
            'anti-hero': 2.2,      # Increased from 2.0
            'moral-ambiguity': 2.4, # Increased from 2.2
            'underdog': 2.1,       # Increased from 1.8
            'rivalry': 2.0,        # Increased from 1.8
            
            # Setting and Theme
            'academy': 2.0,        # Increased from 1.5
            'fantasy-kingdom': 1.8, # Increased from 1.5
            'supernatural-romance': 2.2, # New high priority
            'psychological': 2.1,   # New high priority
            
            # Story Elements
            'mystery': 1.8,        # Increased from 1.5
            'adventure': 1.7,      # Increased from 1.5
            'action': 1.8,         # Increased from 1.5
            'drama': 1.7,          # Increased from 1.5
            
            # Character Relationships
            'found-family': 2.0,   # New high priority
            'mentor-student': 2.1,  # New high priority
            'protective': 2.0,      # New high priority
            'loyalty': 1.9,        # New high priority
            
            # Emotional Elements
            'trauma-healing': 2.2,  # New high priority
            'redemption': 2.1,      # New high priority
            'angst': 1.9,          # New high priority
            'emotional-growth': 2.0 # New high priority
        }
        
        # Enhanced anime matching patterns
        self.anime_patterns = {
            'naruto': ['naruto', 'ninja', 'shinobi', 'chakra', 'rasengan', 'shadow clone'],
            'dragon ball': ['dragon ball', 'ki', 'saiyan', 'super saiyan', 'z-warriors'],
            'jujutsu kaisen': ['jujutsu', 'cursed energy', 'sorcerer', 'curse', 'domain expansion'],
            'demon slayer': ['demon slayer', 'breathing technique', 'nichirin', 'demon', 'hashira'],
            're:zero': ['re:zero', 'return by death', 'subaru', 'emilia', 'witch'],
            'my hero academia': ['my hero academia', 'quirk', 'hero', 'villain', 'all might'],
            'chainsaw man': ['chainsaw man', 'devil', 'contract', 'denji', 'makima'],
            'steins;gate': ['steins;gate', 'time travel', 'mad scientist', 'lab', 'd-mail'],
            'higurashi': ['higurashi', 'when they cry', 'hinamizawa', 'curse', 'looper'],
            'genshin impact': ['genshin impact', 'vision', 'element', 'archon', 'teyvat']
        }

    def _calculate_story_score(self, story: Story, user_profile: UserProfile) -> float:
        """Calculate a score for how well a story matches a user's profile"""
        score = 0.0
        
        # Enhanced anime score calculation
        anime_score = 0.0
        for anime in user_profile.favorite_anime:
            anime = anime.lower()
            if anime in self.anime_patterns:
                for pattern in self.anime_patterns[anime]:
                    if pattern in story.title.lower() or pattern in story.intro.lower():
                        anime_score += 7.0  # Increased from 6.0
                        break
        
        # Enhanced preference matching
        preference_score = 0.0
        for preference in user_profile.preferences.lower().split(','):
            preference = preference.strip()
            if preference in story.title.lower() or preference in story.intro.lower():
                preference_score += 3.5  # Increased from 3.0
                
        # Enhanced interest matching
        interest_score = 0.0
        for interest in user_profile.interests:
            if interest.lower() in story.title.lower() or interest.lower() in story.intro.lower():
                interest_score += 2.5  # Increased from 2.0
                
        # Enhanced tag matching with weights
        tag_score = 0.0
        for tag in story.tags:
            tag = tag.lower()
            if tag in user_profile.preferred_tags:
                tag_score += self.base_tag_weights.get(tag, 1.5)  # Increased base weight
                
        # Enhanced combination bonuses
        if preference_score > 0 and interest_score > 0:
            score += (preference_score + interest_score) * 1.4  # Increased from 1.3
        if preference_score > 0 and tag_score > 0:
            score += (preference_score + tag_score) * 1.3  # Increased from 1.2
        if interest_score > 0 and tag_score > 0:
            score += (interest_score + tag_score) * 1.2  # Increased from 1.1
            
        # Enhanced multiple match bonuses
        if preference_score > 0:
            score += preference_score * 1.3  # Increased from 1.2
        if interest_score > 0:
            score += interest_score * 1.2  # Increased from 1.1
        if tag_score > 0:
            score += tag_score * 1.1  # Increased from 1.0
            
        # Enhanced anime and theme combination bonus
        if anime_score > 0 and (preference_score > 0 or interest_score > 0):
            score += anime_score * 1.4  # Increased from 1.3
            
        # Enhanced exact preference match bonus
        for preference in user_profile.preferences.lower().split(','):
            preference = preference.strip()
            if preference in story.title.lower():
                score += 4.0  # Increased from 3.0
                
        return score
    
    def get_recommendations(self, user_profile: UserProfile, prompt: str = None, num_recommendations: int = 10) -> List[str]:
        """
        Get story recommendations for a user profile using GPT-3.5-turbo
        """
        # Prepare the stories for recommendation
        stories_str = "\n".join([
            f"ID: {story.id}\nTitle: {story.title}\nIntro: {story.intro[:200]}...\nTags: {', '.join(story.tags)}\n"
            for story in self.stories
        ])
        
        user_profile_str = f"""
        User Profile:
        Preferences: {user_profile.preferences}
        Interests: {', '.join(user_profile.interests)}
        Favorite Anime: {', '.join(user_profile.favorite_anime)}
        Preferred Tags: {', '.join(user_profile.preferred_tags)}
        """
        
        # Create recommendation prompt
        recommendation_prompt = f"""
        You are an expert story recommendation specialist. Your task is to select the most relevant stories for a user based on their profile.
        
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
            print(f"\nSending request to GPT-3.5-turbo with {len(self.stories)} stories...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert story recommendation specialist. Return ONLY the story IDs in a comma-separated list."},
                    {"role": "user", "content": recommendation_prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent results
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
            
            # Remove duplicates while preserving order
            seen = set()
            unique_ids = []
            for id in valid_ids:
                if id not in seen:
                    seen.add(id)
                    unique_ids.append(id)
            
            # If we don't have enough recommendations, fall back to scoring-based recommendations
            if len(unique_ids) < num_recommendations:
                print(f"Warning: GPT only returned {len(unique_ids)} valid recommendations. Falling back to scoring-based recommendations.")
                story_scores = []
                for story in self.stories:
                    score = self._calculate_story_score(story, user_profile)
                    story_scores.append((story, score))
                
                story_scores.sort(key=lambda x: x[1], reverse=True)
                fallback_ids = [s.id for s, _ in story_scores[:num_recommendations]]
                
                # Combine GPT recommendations with fallback recommendations
                combined_ids = unique_ids + [id for id in fallback_ids if id not in unique_ids]
                return combined_ids[:num_recommendations]
            
            return unique_ids[:num_recommendations]
            
        except Exception as e:
            print(f"Error in GPT recommendation generation: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"API Response: {e.response}")
            
            # Fallback to scoring-based recommendations if GPT fails
            print("Falling back to scoring-based recommendations...")
            story_scores = []
            for story in self.stories:
                score = self._calculate_story_score(story, user_profile)
                story_scores.append((story, score))
            
            story_scores.sort(key=lambda x: x[1], reverse=True)
            return [s.id for s, _ in story_scores[:num_recommendations]] 