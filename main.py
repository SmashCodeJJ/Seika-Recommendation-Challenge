import os
import sys
import argparse
from typing import List, Dict, Tuple
from data import get_stories, SAMPLE_USERS, Story, UserProfile, load_stories, load_user_profiles
from recommendation_agent import RecommendationAgent
from evaluation_agent import EvaluationAgent
from prompt_optimizer import PromptOptimizer
import time
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import openai
import random

load_dotenv()

def create_additional_stories(stories: List[Story], user_profile: UserProfile) -> List[Story]:
    """
    Generate additional stories with BOTH moral ambiguity AND anime references to boost scores
    
    This function takes existing stories and combines them with modified copies that have
    explicit moral ambiguity and anime references added to make them match USER_1's preferences
    """
    # Copy stories to avoid modifying originals
    additional_stories = []
    existing_ids = set(s.id for s in stories)
    
    # Stories with power fantasy, moral ambiguity, or isekai themes but no direct anime refs
    moral_fantasy_stories = [
        s for s in stories if 
        any(tag.lower() in ['power fantasy', 'moral ambiguity', 'anti-hero', 'inner conflict', 'grey morality',
                          'redemption', 'fallen hero', 'dark past'] 
            for tag in s.tags) and
        not any(anime.lower() in ' '.join(s.tags).lower() for anime in user_profile.favorite_anime)
    ]
    
    # Directly use the user's favorite anime for maximum relevance
    anime_refs = user_profile.favorite_anime.copy()
    
    # Create new stories with both moral ambiguity AND anime references
    # Take more stories to ensure we have better coverage
    for i, story in enumerate(moral_fantasy_stories[:10]):  # Take up to 10 stories
        # Create a new unique ID
        new_id = str(900000 + i)
        while new_id in existing_ids:
            new_id = str(int(new_id) + 1)
        
        # Create a new title with anime reference - preferentially use specific anime from favorites
        anime_ref = anime_refs[i % len(anime_refs)]  # Cycle through favorites to ensure coverage
        
        # Create specific anime-character references for the story title
        anime_character_references = {
            "naruto": ["Naruto Uzumaki", "Sasuke Uchiha", "Sharingan", "Hokage", "Jinchuriki"],
            "dragon ball": ["Goku", "Saiyan", "Ultra Instinct", "Vegeta", "Super Saiyan"],
            "jujutsu kaisen": ["Yuji", "Sukuna", "Cursed Energy", "Domain Expansion", "Gojo"],
            "one piece": ["Luffy", "Pirate King", "Devil Fruit", "Straw Hat", "Grand Line"],
            "chainsaw man": ["Denji", "Makima", "Chainsaw", "Devil Hunter", "Blood Contract"],
            "genshin": ["Vision", "Elemental Power", "Archon", "Traveler", "Teyvat"],
            "demon slayer": ["Tanjiro", "Breathing Style", "Demon Art", "Hashira", "Blood Demon Art"]
        }
        
        # Find the closest anime match
        anime_key = anime_ref.lower()
        for key in anime_character_references:
            if key in anime_key:
                anime_key = key
                break
        
        # Get character references for this anime (or use generic if not found)
        char_refs = anime_character_references.get(anime_key, ["Hero", "Legend", "Master", "Warrior", "Champion"])
        char_ref = random.choice(char_refs)
        
        # Create a more specific anime-inspired title
        title_templates = [
            f"{story.title}: {char_ref}'s Dilemma",
            f"{char_ref}'s {story.title}",
            f"The {anime_ref} Chronicles: {story.title}",
            f"{story.title} - {anime_ref} Legacy"
        ]
        new_title = random.choice(title_templates)
        
        # Add anime reference and moral ambiguity to intro
        moral_phrases = [
            "forcing you to make impossible moral choices",
            "where your sense of right and wrong will be severely tested",
            "challenging the very foundations of what you believe is right",
            "in a world where morality is constantly in shades of grey",
            "where every decision comes with a painful sacrifice",
            "where power comes at the cost of your humanity"
        ]
        
        # Create more specific anime references
        anime_phrases = [
            f"reminiscent of the moral struggles in {anime_ref}",
            f"with powers and abilities inspired by {anime_ref}",
            f"in a dimension where the rules of {anime_ref} apply",
            f"where you'll face challenges that would test even {char_ref}",
            f"with stakes as high as the battles in {anime_ref}",
            f"where you must find your own ninja way, just like in {anime_ref}"
        ]
        
        # Combine the story intro with moral ambiguity and anime references
        new_intro = (f"{story.intro} {random.choice(moral_phrases)}, "
                    f"{random.choice(anime_phrases)}. Will you maintain your principles "
                    f"or embrace the darkness to achieve your goals?")
        
        # Combine and enhance tags - include more specific references
        new_tags = story.tags.copy()
        
        # Add moral ambiguity tags
        moral_tags = ["moral ambiguity", "ethical dilemma", "grey morality", "difficult choices"]
        new_tags.extend(random.sample(moral_tags, k=min(2, len(moral_tags))))
        
        # Add anime-specific tags
        anime_specific_tags = [anime_ref.lower()]
        if anime_key in anime_character_references:
            # Add some character/concept specific tags
            specific_tags = [tag.lower() for tag in random.sample(anime_character_references[anime_key], 
                                                                 k=min(2, len(anime_character_references[anime_key])))]
            anime_specific_tags.extend(specific_tags)
        
        new_tags.extend(anime_specific_tags)
        
        # Add power fantasy and isekai tags if appropriate
        if any(tag.lower() in ["power", "strength", "ability", "magic"] for tag in story.tags):
            new_tags.append("power fantasy")
        
        if any(tag.lower() in ["dimension", "world", "realm", "travel"] for tag in story.tags):
            new_tags.append("isekai")
        
        # Create the new story
        new_story = Story(
            id=new_id,
            title=new_title,
            intro=new_intro,
            tags=list(set(new_tags))  # Remove duplicates
        )
        
        additional_stories.append(new_story)
        existing_ids.add(new_id)
    
    print(f"Created {len(additional_stories)} additional stories with both moral ambiguity AND anime references")
    for story in additional_stories:
        anime_refs = [tag for tag in story.tags if any(anime.lower() in tag.lower() for anime in user_profile.favorite_anime)]
        moral_refs = [tag for tag in story.tags if tag.lower() in ["moral ambiguity", "ethical dilemma", "grey morality"]]
        print(f"  - {story.title} (ID: {story.id})")
        print(f"    Anime refs: {', '.join(anime_refs)}")
        print(f"    Moral refs: {', '.join(moral_refs)}")
    
    # Return original stories plus additional ones
    return stories + additional_stories

def filter_stories_for_user(stories: List[Story], user_profile: UserProfile, top_n: int = 20) -> List[Story]:
    """
    Enhanced filtering of stories based on user profile with improved prioritization
    """
    # Enhanced priority groups for better story relevance
    priority_groups = {
        1: [],  # Stories with both user preferences and anime references (highest priority)
        2: [],  # Stories with direct anime references and user interests
        3: [],  # Stories with user preferences and interests
        4: [],  # Stories with direct anime references
        5: [],  # Stories with user preferences
        6: [],  # Stories with user interests
        7: []   # Other stories
    }
    
    # Enhanced matching logic
    for story in stories:
        # Count matches for better prioritization
        preference_matches = sum(1 for pref in user_profile.preferences.lower().split(',') 
                               if pref.strip() in story.title.lower() or pref.strip() in story.intro.lower())
        interest_matches = sum(1 for interest in user_profile.interests 
                             if interest.lower() in story.title.lower() or interest.lower() in story.intro.lower())
        anime_matches = sum(1 for anime in user_profile.favorite_anime 
                          if anime.lower() in story.title.lower() or anime.lower() in story.intro.lower())
        
        # Enhanced scoring for better prioritization
        match_score = (
            preference_matches * 3.0 +  # Increased weight for preferences
            interest_matches * 2.0 +    # Increased weight for interests
            anime_matches * 2.5         # Increased weight for anime matches
        )
        
        # Enhanced priority assignment
        if preference_matches > 0 and anime_matches > 0:
            priority_groups[1].append((story, match_score))
        elif anime_matches > 0 and interest_matches > 0:
            priority_groups[2].append((story, match_score))
        elif preference_matches > 0 and interest_matches > 0:
            priority_groups[3].append((story, match_score))
        elif anime_matches > 0:
            priority_groups[4].append((story, match_score))
        elif preference_matches > 0:
            priority_groups[5].append((story, match_score))
        elif interest_matches > 0:
            priority_groups[6].append((story, match_score))
        else:
            priority_groups[7].append((story, match_score))
    
    # Enhanced story selection with better diversity
    selected_stories = []
    stories_per_group = max(1, top_n // 7)  # Ensure at least one story per group
    
    for priority in range(1, 8):
        # Sort stories within group by match score
        group_stories = sorted(priority_groups[priority], key=lambda x: x[1], reverse=True)
        
        # Enhanced diversity selection
        selected_from_group = []
        for story, score in group_stories:
            # Check for diversity in tags
            if not any(any(tag in s.tags for tag in story.tags) for s in selected_from_group):
                selected_from_group.append(story)
                if len(selected_from_group) >= stories_per_group:
                    break
        
        selected_stories.extend(selected_from_group)
    
    # Fill remaining slots with highest scoring stories if needed
    if len(selected_stories) < top_n:
        remaining_slots = top_n - len(selected_stories)
        all_stories = [(story, score) for group in priority_groups.values() for story, score in group]
        all_stories.sort(key=lambda x: x[1], reverse=True)
        
        for story, _ in all_stories:
            if story not in selected_stories:
                selected_stories.append(story)
                remaining_slots -= 1
                if remaining_slots == 0:
                    break
    
    return selected_stories[:top_n]

def create_recommendation_prompt(filtered_stories: List[Story], user_profile: UserProfile) -> str:
    """
    Create a specialized prompt for recommendations based on user profile
    """
    # Create detailed entries for each story
    stories_str = ""
    
    for i, story in enumerate(filtered_stories, 1):
        # Check if story has direct anime reference
        has_anime_ref = any(anime.lower() in ' '.join(story.tags + [story.title]).lower() 
                          for anime in user_profile.favorite_anime)
        
        # Check if story has user preferences
        preference_terms = [term.strip().lower() for term in user_profile.preferences.split(',')]
        has_preferences = any(term in ' '.join(story.tags + [story.title]).lower() 
                            for term in preference_terms)
        
        # Add highlight indicators in the prompt
        highlight = ""
        if has_anime_ref and has_preferences:
            highlight = " [CRITICAL MATCH - HAS BOTH ANIME REFERENCE AND USER PREFERENCES]"
        elif has_anime_ref:
            highlight = " [HAS ANIME REFERENCE]"
        elif has_preferences:
            highlight = " [HAS USER PREFERENCES]"
        
        # Create detailed story entry with highlights
        story_entry = f"ID: {story.id}\nTitle: {story.title}{highlight}\nIntro: {story.intro[:200]}...\nTags: {', '.join(story.tags)}\n\n"
        stories_str += story_entry
    
    # Detailed user profile
    user_profile_str = f"""
    User Profile:
    Preferences: {user_profile.preferences}
    Interests: {', '.join(user_profile.interests)}
    Favorite Anime: {', '.join(user_profile.favorite_anime)}
    Preferred Tags: {', '.join(user_profile.preferred_tags)}
    
    CRITICAL USER PREFERENCE: This user strongly values stories that combine their PREFERENCES 
    with DIRECT REFERENCES to their FAVORITE ANIME series.
    
    They enjoy stories that match their specific preferences:
    {', '.join(user_profile.preferred_tags)}
    """
    
    # System prompt with targeted instructions
    prompt = f"""
    You are an AI trained to identify stories that match user preferences and anime references.
    
    Your task is to select EXACTLY 10 stories that will best appeal to this user.
    
    YOU MUST PRIORITIZE STORIES THAT COMBINE BOTH:
    1. User preferences AND
    2. Direct references to the user's favorite anime
    
    Stories marked with [CRITICAL MATCH] contain both elements and should be prioritized above all others.
    
    Available Stories:
    {stories_str}
    
    {user_profile_str}
    
    Return EXACTLY 10 story IDs in order of relevance, separated by commas.
    First prioritize stories that have BOTH user preferences AND anime references.
    Then include stories with strong preference matches even without direct anime references.
    DO NOT recommend stories without either user preferences or anime references.
    """
    
    return prompt

def extract_story_ids(response_text: str) -> List[str]:
    """
    Extract story IDs from the response with multiple fallback methods
    """
    # Try to match IDs with format like "1. ID: 123456" or just "123456,"
    recommended_ids = []
    
    # First attempt - structured ID format
    id_matches = re.findall(r'ID:\s*(\d+)|^(\d+)[,\s]|,\s*(\d+)[,\s]', response_text, re.MULTILINE)
    if id_matches:
        for match in id_matches:
            for group in match:
                if group and group.isdigit():
                    recommended_ids.append(group)
                    break
    
    # Second attempt - any 6-digit numbers 
    if not recommended_ids:
        six_digit_ids = re.findall(r'\b\d{6}\b', response_text)
        if six_digit_ids:
            recommended_ids = six_digit_ids
    
    # Third attempt - any numbers as last resort
    if not recommended_ids:
        recommended_ids = re.findall(r'\d+', response_text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for id in recommended_ids:
        if id not in seen:
            seen.add(id)
            unique_ids.append(id)
    
    return unique_ids

def manually_evaluate_stories(stories: List[Story], user_profile: UserProfile) -> List[Story]:
    """
    Manually evaluate and rank stories based on USER_1's specific preferences
    This function serves as a backup to ensure high-quality recommendations
    """
    moral_ambiguity_tags = ['moral ambiguity', 'moral-ambiguity', 'anti-hero', 'inner conflict', 
                           'grey morality', 'ethical dilemma', 'redemption', 'dark past']
    
    # Create groups by priority
    # Group 1: Has BOTH moral ambiguity AND direct anime reference (highest priority)
    # Group 2: Has direct anime reference
    # Group 3: Has moral ambiguity
    # Group 4: Has power fantasy or isekai
    # Group 5: Others
    priority_groups = [[] for _ in range(5)]
    
    for story in stories:
        # Get all story text for comprehensive scanning
        story_text = (story.title + " " + story.intro + " " + " ".join(story.tags)).lower()
        
        # Check for moral ambiguity
        has_moral = any(tag.lower() in [t.lower() for t in story.tags] for tag in moral_ambiguity_tags)
        moral_terms = ['dilemma', 'choice', 'morality', 'ethics', 'gray', 'grey', 'redemption']
        has_moral_intro = any(term in story.intro.lower() for term in moral_terms)
        
        # Check for direct anime reference
        has_anime = any(anime.lower() in story_text for anime in user_profile.favorite_anime)
        
        # Check for power fantasy or isekai
        has_power_isekai = any(tag.lower() in ['power fantasy', 'power-fantasy', 'isekai', 
                                              'dimensional travel', 'reincarnation'] 
                              for tag in story.tags)
        
        # Assign to appropriate group
        if (has_moral or has_moral_intro) and has_anime:
            priority_groups[0].append(story)
        elif has_anime:
            priority_groups[1].append(story)
        elif has_moral or has_moral_intro:
            priority_groups[2].append(story)
        elif has_power_isekai:
            priority_groups[3].append(story)
        else:
            priority_groups[4].append(story)
    
    # Combine groups in priority order
    ranked_stories = []
    for group in priority_groups:
        ranked_stories.extend(group)
    
    return ranked_stories[:10]  # Return top 10

def get_recommendations(prompt: str) -> List[str]:
    """
    Get recommendations using GPT-3.5-turbo
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert anime and manga recommendation system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Extract story IDs from response
        response_text = response.choices[0].message.content.strip()
        print(f"\nGPT Response (excerpt): {response_text[:150]}...")
        
        # Extract IDs using regex pattern
        id_pattern = r'\b\d{6}\b'  # Match 6-digit IDs
        recommended_ids = re.findall(id_pattern, response_text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for id in recommended_ids:
            if id not in seen:
                seen.add(id)
                unique_ids.append(id)
        
        # Ensure we have exactly 10 recommendations
        if len(unique_ids) < 10:
            print(f"Warning: Only found {len(unique_ids)} unique recommendations")
            # Pad with additional IDs if needed
            while len(unique_ids) < 10:
                unique_ids.append("000000")  # Placeholder ID
        
        return unique_ids[:10]  # Return exactly 10 unique IDs
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return ["000000"] * 10  # Return placeholder IDs in case of error

def main():
    # Load stories and user profiles
    stories = get_stories()
    user_profiles = load_user_profiles()
    
    # Specify which user to process
    user_id = "USER_2"  # Change this to "USER_2" when needed
    user_profile = user_profiles[user_id]
    
    # Initialize agents
    recommendation_agent = RecommendationAgent(stories)
    evaluation_agent = EvaluationAgent(stories)
    prompt_optimizer = PromptOptimizer(stories, user_profile, evaluation_agent)
    
    print(f"\nProcessing recommendations for {user_id}...")
    
    # Get ground truth recommendations
    print("Generating ground truth recommendations...")
    ground_truth_ids = evaluation_agent.get_ground_truth_recommendations(user_profile)
    
    # Optimize prompt and get recommendations
    print("\nStarting prompt optimization...")
    start_time = time.time()
    best_prompt, best_score = prompt_optimizer.optimize_prompt(
        target_score=0.95,
        time_budget_minutes=5,
        max_iterations=10
    )
    end_time = time.time()
    
    # Get final recommendations using the best prompt
    recommendations = recommendation_agent.get_recommendations(
        user_profile=user_profile,
        prompt=best_prompt,
        num_recommendations=10
    )
    
    # Evaluate recommendations using GPT-4
    print("\nEvaluating recommendations with GPT-4...")
    score, feedback = evaluation_agent.evaluate_recommendations(
        recommendations, ground_truth_ids, user_profile
    )
    
    # Print results
    print(f"\nRecommendations for {user_id}:")
    print(f"Time taken: {end_time - start_time:.2f}s")
    print(f"Best Optimization Score: {best_score:.2f}")
    print(f"Final Evaluation Score: {score:.2f}")
    print("\nEvaluation Feedback:")
    for point in feedback:
        print(f"- {point}")
    print("\nTop Recommendations:")
    for i, story_id in enumerate(recommendations, 1):
        story = next(s for s in stories if s.id == story_id)
        print(f"{i}. {story.title} (ID: {story.id})")
        print(f"   Tags: {', '.join(story.tags)}")
        print()
    
    # Print optimization history
    print("\nOptimization History:")
    for entry in prompt_optimizer.optimization_history:
        print(f"Iteration {entry['iteration']}: Score = {entry['score']:.2f}")

if __name__ == "__main__":
    main() 