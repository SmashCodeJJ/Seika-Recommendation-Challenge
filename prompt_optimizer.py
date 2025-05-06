import os
from typing import List, Dict, Tuple
import openai
from dotenv import load_dotenv
import random
import numpy as np
from data import Story, UserProfile
from evaluation_agent import EvaluationAgent

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class PromptOptimizer:
    def __init__(self, stories: List[Story], user_profile: UserProfile, evaluation_agent: EvaluationAgent):
        self.stories = stories
        self.user_profile = user_profile
        self.evaluation_agent = evaluation_agent
        self.best_prompt = None
        self.best_score = 0
        self.optimization_history = []
        self.successful_patterns = []
        
        # Define prompt components with weights
        self.prompt_components = {
            'context': [
                ("You are a story recommendation system that matches stories to user preferences.", 1.0),
                ("You are an expert anime and manga recommendation system.", 1.0),
                ("You are a personalized story recommendation engine.", 1.0)
            ],
            'instruction': [
                ("Recommend stories that match the user's preferences and interests.", 1.0),
                ("Find stories that align with the user's favorite anime and preferred tags.", 1.0),
                ("Select stories that would appeal to the user based on their profile.", 1.0)
            ],
            'format': [
                ("Return only the story IDs in a comma-separated list, ordered by relevance.", 1.0),
                ("Provide a list of story IDs, most relevant first, separated by commas.", 1.0),
                ("List the story IDs in order of relevance, separated by commas.", 1.0)
            ],
            'emphasis': [
                ("Pay special attention to the user's favorite anime and preferred tags.", 1.0),
                ("Focus on matching the user's interests and preferred story elements.", 1.0),
                ("Prioritize stories that align with the user's preferences and interests.", 1.0)
            ]
        }
        
    def generate_prompt(self, components: Dict[str, Tuple[str, float]]) -> str:
        """Generate a prompt using selected components with their weights"""
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
        {components['context'][0]}
        
        {components['instruction'][0]}
        {components['emphasis'][0]}
        
        Stories:
        {stories_str}
        
        User Profile:
        {user_profile_str}
        
        {components['format'][0]}
        """
    
    def mutate_prompt_components(self, components: Dict[str, Tuple[str, float]]) -> Dict[str, Tuple[str, float]]:
        """Create a new set of components by mutating the current ones with weighted selection"""
        new_components = components.copy()
        
        # Randomly select a component to mutate
        component_to_mutate = random.choice(list(self.prompt_components.keys()))
        
        # Get a new value for the selected component using weighted selection
        weights = [w for _, w in self.prompt_components[component_to_mutate]]
        values = [v for v, _ in self.prompt_components[component_to_mutate]]
        
        # Adjust weights based on successful patterns
        for pattern in self.successful_patterns:
            if pattern['component'] == component_to_mutate:
                for i, value in enumerate(values):
                    if value == pattern['value']:
                        weights[i] *= 1.2  # Boost successful patterns
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Select new value based on weights
        new_value = random.choices(values, weights=weights, k=1)[0]
        while new_value == components[component_to_mutate][0]:
            new_value = random.choices(values, weights=weights, k=1)[0]
        
        new_components[component_to_mutate] = (new_value, 1.0)
        return new_components
    
    def update_component_weights(self, components: Dict[str, Tuple[str, float]], score: float):
        """Update component weights based on performance"""
        for component, (value, weight) in components.items():
            # Find the index of the current value
            values = [v for v, _ in self.prompt_components[component]]
            if value in values:
                idx = values.index(value)
                # Update weight based on score
                self.prompt_components[component][idx] = (value, weight * (1 + score))
    
    def optimize_prompt(self, target_score: float, time_budget_minutes: int, max_iterations: int = 20) -> Tuple[str, float]:
        """Optimize the prompt using a genetic algorithm approach with feedback loop"""
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
                
                # Get ground truth recommendations
                ground_truth_ids = self.evaluation_agent.get_ground_truth_recommendations(
                    self.user_profile, num_recommendations=10
                )
                
                # Evaluate recommendations
                score, feedback = self.evaluation_agent.evaluate_recommendations(
                    recommended_ids, ground_truth_ids, self.user_profile
                )
                
                # Update if better
                if score > current_score:
                    current_score = score
                    current_components = new_components
                    current_prompt = new_prompt
                    
                    # Update best if better
                    if current_score > self.best_score:
                        self.best_score = current_score
                        self.best_prompt = current_prompt
                        
                        # Record successful patterns
                        for component, (value, _) in current_components.items():
                            self.successful_patterns.append({
                                'component': component,
                                'value': value,
                                'score': current_score
                            })
                    
                    # Update component weights
                    self.update_component_weights(current_components, score)
                
                # Record optimization history
                self.optimization_history.append({
                    'iteration': iteration,
                    'score': current_score,
                    'prompt': current_prompt,
                    'feedback': feedback
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