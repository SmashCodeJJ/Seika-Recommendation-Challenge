import os
import sys
import argparse
from typing import List
from data import get_stories, SAMPLE_USERS
from recommendation_agent import RecommendationAgent
from evaluation_agent import EvaluationAgent
from prompt_optimizer import PromptOptimizer
import time
import json
from datetime import datetime

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate story recommendations for a specific user profile.')
    parser.add_argument('--user', type=str, default='USER_1', 
                       choices=['USER_1', 'USER_2', 'USER_3'],
                       help='User profile to generate recommendations for (default: USER_1)')
    args = parser.parse_args()
    
    # Get the selected user profile
    user_profile = SAMPLE_USERS[args.user]
    print(f"\nGenerating recommendations for {args.user}")
    print(f"Preferences: {user_profile.preferences}")
    print(f"Interests: {', '.join(user_profile.interests)}")
    print(f"Favorite Anime: {', '.join(user_profile.favorite_anime)}")
    print(f"Preferred Tags: {', '.join(user_profile.preferred_tags)}\n")
    
    # Load stories
    stories = get_stories()
    print(f"Loaded {len(stories)} stories")
    
    # Initialize agents
    evaluation_agent = EvaluationAgent(stories)
    recommendation_agent = RecommendationAgent(stories)
    prompt_optimizer = PromptOptimizer(stories, user_profile)
    
    # Get ground truth recommendations
    print("\nGenerating ground truth recommendations...")
    ground_truth_ids = evaluation_agent.get_ground_truth_recommendations(user_profile)
    print(f"Ground truth recommendations: {ground_truth_ids}")
    
    # Optimization parameters
    TARGET_SCORE = 0.9  # Increased target score
    TIME_BUDGET_MINUTES = 15  # Increased time budget
    MAX_ITERATIONS = 25  # Increased max iterations
    
    print(f"\nStarting optimization with:")
    print(f"Target score: {TARGET_SCORE}")
    print(f"Time budget: {TIME_BUDGET_MINUTES} minutes")
    print(f"Max iterations: {MAX_ITERATIONS}")
    
    # Start optimization
    start_time = time.time()
    best_prompt, best_score = prompt_optimizer.optimize_prompt(
        target_score=TARGET_SCORE,
        time_budget_minutes=TIME_BUDGET_MINUTES,
        max_iterations=MAX_ITERATIONS
    )
    
    # Get final recommendations using best prompt
    print("\nGetting final recommendations...")
    final_recommendations = recommendation_agent.get_recommendations(user_profile, best_prompt)
    final_score, feedback = evaluation_agent.evaluate_recommendations(
        final_recommendations, 
        ground_truth_ids,
        user_profile=user_profile
    )
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best score achieved: {best_score:.2f}")
    print(f"Final score: {final_score:.2f}")
    print("\nFinal recommendations:")
    for story_id in final_recommendations:
        story = next(s for s in stories if s.id == story_id)
        print(f"ID: {story.id}, Title: {story.title}")
    
    print("\nFeedback:")
    for item in feedback:
        print(f"- {item}")
    
    # Save optimization history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = f"optimization_history_{args.user}_{timestamp}.json"
    with open(history_file, 'w') as f:
        json.dump({
            'user_profile': args.user,
            'timestamp': timestamp,
            'target_score': TARGET_SCORE,
            'time_budget': TIME_BUDGET_MINUTES,
            'max_iterations': MAX_ITERATIONS,
            'best_score': best_score,
            'final_score': final_score,
            'optimization_history': prompt_optimizer.optimization_history,
            'final_recommendations': final_recommendations,
            'feedback': feedback
        }, f, indent=2)
    
    print(f"\nOptimization history saved to {history_file}")

if __name__ == "__main__":
    main() 