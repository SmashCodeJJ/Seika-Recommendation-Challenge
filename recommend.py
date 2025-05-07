import os
from dotenv import load_dotenv
from data import get_stories, SAMPLE_USER
from evaluation_agent import EvaluationAgent
from prompt_optimizer import PromptOptimizer

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize agents
    stories = get_stories()
    print(f"\nLoaded {len(stories)} stories\n")
    
    evaluation_agent = EvaluationAgent(stories)
    prompt_optimizer = PromptOptimizer()
    
    print("Generating ground truth recommendations...")
    ground_truth_ids = evaluation_agent.get_ground_truth_recommendations(SAMPLE_USER)
    
    # Set optimization parameters
    target_score = 0.95
    time_budget_minutes = 10
    max_iterations = 15
    
    print(f"\nStarting optimization with:")
    print(f"Target score: {target_score}")
    print(f"Time budget: {time_budget_minutes} minutes")
    print(f"Max iterations: {max_iterations}")
    
    # Run optimization
    best_prompt, best_score, final_recommendations = prompt_optimizer.optimize(
        user_profile=SAMPLE_USER,
        evaluation_agent=evaluation_agent,
        ground_truth_ids=ground_truth_ids,
        target_score=target_score,
        time_budget_minutes=time_budget_minutes,
        max_iterations=max_iterations
    )
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best Score: {best_score:.2f}")
    print("\nFinal Recommendations:")
    for story in stories:
        if story.id in final_recommendations:
            print(f"\nTitle: {story.title}")
            print(f"Tags: {', '.join(story.tags)}")

if __name__ == "__main__":
    main() 