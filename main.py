import os
import sys
import argparse
from typing import List
from data import get_stories, SAMPLE_USERS, Story, UserProfile, load_stories, load_user_profiles
from recommendation_agent import RecommendationAgent
from evaluation_agent import EvaluationAgent
from prompt_optimizer import PromptOptimizer
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import openai

load_dotenv()

def main():
    # Load environment variables
    load_dotenv()
    
    # Load stories and user profiles
    stories = load_stories()
    user_profiles = load_user_profiles()
    
    # Initialize evaluation agent
    evaluation_agent = EvaluationAgent(stories)
    
    # Process only USER_1
    user_profile = user_profiles["USER_3"]
            
    print(f"\n{'='*80}")
    print(f"Processing recommendations for USER_3")
    print(f"{'='*80}")
    
    # Get ground truth recommendations
    ground_truth_ids = evaluation_agent.get_ground_truth_recommendations(user_profile)
    
    # Initialize prompt optimizer with evaluation agent
    optimizer = PromptOptimizer(stories, user_profile, evaluation_agent)
    
    # Optimize prompt
    best_prompt, best_score = optimizer.optimize_prompt(
        target_score=0.8,
        time_budget_minutes=5,
        max_iterations=5
    )
    
    print(f"\nBest prompt score: {best_score:.2f}")
    
    # Get final recommendations using the optimized prompt
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": best_prompt},
                {"role": "user", "content": f"User Profile:\n{user_profile.preferences}\nInterests: {', '.join(user_profile.interests)}\nFavorite Anime: {', '.join(user_profile.favorite_anime)}\nPreferred Tags: {', '.join(user_profile.preferred_tags)}\n\nPlease recommend EXACTLY 10 stories."}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract story IDs from response
        recommended_story_ids = []
        try:
            # Try to parse the response as a list of IDs
            response_text = response.choices[0].message.content.strip()
            # Extract numbers from the response
            import re
            recommended_story_ids = [str(id) for id in re.findall(r'\d+', response_text)]  # Convert to strings to match ground truth IDs
        except Exception as e:
            print(f"Error parsing recommendations: {e}")
            return
        
        # Get the actual story objects
        recommended_stories = [story for story in stories if story.id in recommended_story_ids]
        
        # If we have fewer than 10 stories, fill in with top-matching stories
        if len(recommended_stories) < 10:
            # Get all stories not already recommended
            remaining_stories = [s for s in stories if s.id not in recommended_story_ids]
            
            # Calculate match scores for remaining stories
            story_scores = []
            for story in remaining_stories:
                # Calculate tag overlap with user preferences
                tag_overlap = len(set(story.tags) & set(user_profile.preferred_tags))
                # Calculate interest overlap
                interest_overlap = len(set(story.tags) & set(user_profile.interests))
                # Calculate anime reference overlap
                anime_overlap = sum(1 for anime in user_profile.favorite_anime if any(tag in story.tags for tag in anime.lower().split('-')))
                
                # Calculate total score
                score = tag_overlap * 2 + interest_overlap * 1.5 + anime_overlap
                story_scores.append((story, score))
            
            # Sort by score and get top stories to fill up to 10
            story_scores.sort(key=lambda x: x[1], reverse=True)
            additional_stories = [s for s, _ in story_scores[:10 - len(recommended_stories)]]
            
            # Add additional stories to recommendations
            recommended_stories.extend(additional_stories)
        
        # Evaluate final recommendations
        final_score, feedback = evaluation_agent.evaluate_recommendations(
            [s.id for s in recommended_stories],
            ground_truth_ids,
            user_profile
        )
        
        print(f"\nFinal Recommendation Score: {final_score:.2f}")
        print("\nHighly Recommended Stories:")
        for story in recommended_stories[:10]:  # Show top 10 recommendations
            print(f"\nID: {story.id}")
            print(f"Title: {story.title}")
            print(f"Tags: {', '.join(story.tags)}")
        
        print("\nEvaluation Feedback:")
        print(feedback)
        
        print("\nSuccessful Patterns Learned:")
        for pattern in optimizer.successful_patterns:
            print(f"- {pattern}")
            
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return

if __name__ == "__main__":
    main() 