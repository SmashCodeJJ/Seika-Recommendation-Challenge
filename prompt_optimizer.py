import os
from typing import List, Dict, Tuple
import openai
from dotenv import load_dotenv
import random
import numpy as np
from data import Story, UserProfile

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class PromptOptimizer:
    def __init__(self, stories: List[Story], user_profile: UserProfile):
        self.stories = stories
        self.user_profile = user_profile
        self.best_prompt = None
        self.best_score = 0
        self.optimization_history = []
        
        # Define prompt components
        self.prompt_components = {
            'context': [
                "You are a story recommendation system that matches stories to user preferences.",
                "You are an expert anime and manga recommendation system.",
                "You are a personalized story recommendation engine."
            ],
            'instruction': [
                "Recommend stories that match the user's preferences and interests.",
                "Find stories that align with the user's favorite anime and preferred tags.",
                "Select stories that would appeal to the user based on their profile."
            ],
            'format': [
                "Return only the story IDs in a comma-separated list, ordered by relevance.",
                "Provide a list of story IDs, most relevant first, separated by commas.",
                "List the story IDs in order of relevance, separated by commas."
            ],
            'emphasis': [
                "Pay special attention to the user's favorite anime and preferred tags.",
                "Focus on matching the user's interests and preferred story elements.",
                "Prioritize stories that align with the user's preferences and interests."
            ]
        }
        
    def generate_prompt(self, components: Dict[str, str]) -> str:
        """Generate a prompt using selected components"""
        stories_str = "\n".join([
            f"ID: {story.id}\nTitle: {story.title}\nTags: {', '.join(story.tags)}\n"
            for story in self.stories
        ])
        
        user_profile_str = f"""
        User Preferences: {self.user_profile.preferences}
        Interests: {', '.join(self.user_profile.interests)}
        Favorite Anime: {', '.join(self.user_profile.favorite_anime)}
        Preferred Tags: {', '.join(self.user_profile.preferred_tags)}
        """
        
        return f"""
        {components['context']}
        
        {components['instruction']}
        {components['emphasis']}
        
        Stories:
        {stories_str}
        
        User Profile:
        {user_profile_str}
        
        {components['format']}
        """
    
    def mutate_prompt_components(self, components: Dict[str, str]) -> Dict[str, str]:
        """Create a new set of components by mutating the current ones"""
        new_components = components.copy()
        
        # Randomly select a component to mutate
        component_to_mutate = random.choice(list(self.prompt_components.keys()))
        
        # Get a new value for the selected component
        new_value = random.choice(self.prompt_components[component_to_mutate])
        while new_value == components[component_to_mutate]:
            new_value = random.choice(self.prompt_components[component_to_mutate])
        
        new_components[component_to_mutate] = new_value
        return new_components
    
    def optimize_prompt(self, target_score: float, time_budget_minutes: int, max_iterations: int = 20) -> Tuple[str, float]:
        """Optimize the prompt using a genetic algorithm approach"""
        # Initialize with random components
        current_components = {
            key: random.choice(values)
            for key, values in self.prompt_components.items()
        }
        
        current_prompt = self.generate_prompt(current_components)
        current_score = 0
        
        for iteration in range(max_iterations):
            # Generate new prompt by mutating components
            new_components = self.mutate_prompt_components(current_components)
            new_prompt = self.generate_prompt(new_components)
            
            # Get recommendations using the new prompt
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": new_prompt},
                        {"role": "user", "content": "Please recommend 10 stories."}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                recommended_ids = response.choices[0].message.content.strip().split(',')
                recommended_ids = [id.strip() for id in recommended_ids if id.strip().isdigit()]
                
                # Calculate score (placeholder - actual score should come from evaluation)
                new_score = len(recommended_ids) / 10  # Simple metric for demonstration
                
                # Update if better
                if new_score > current_score:
                    current_score = new_score
                    current_components = new_components
                    current_prompt = new_prompt
                    
                    # Update best if better
                    if current_score > self.best_score:
                        self.best_score = current_score
                        self.best_prompt = current_prompt
                
                # Record optimization history
                self.optimization_history.append({
                    'iteration': iteration,
                    'score': current_score,
                    'prompt': current_prompt
                })
                
                # Early stopping if target score is reached
                if current_score >= target_score:
                    break
                    
            except Exception as e:
                print(f"Error in iteration {iteration}: {str(e)}")
                continue
        
        return self.best_prompt, self.best_score

    def should_continue_optimization(self, current_score: float, max_iterations: int = 5) -> bool:
        """
        Determine if optimization should continue based on score history and iteration count
        """
        if len(self.optimization_history) >= max_iterations:
            return False
            
        if len(self.optimization_history) >= 2:
            # Check if the score has plateaued
            recent_scores = [history['score'] for history in self.optimization_history[-2:]]
            if abs(recent_scores[1] - recent_scores[0]) < 0.05:
                return False
                
        return True 