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
        # Base weights will be dynamically generated based on user profile
        self.base_tag_weights = {}
        self.anime_weights = {}
        self.moral_ambiguity_markers = [
            'conflicted', 'dilemma', 'choice', 'morality', 'ethics', 'gray', 'grey',
            'right and wrong', 'good and evil', 'lesser evil', 'sacrifice', 'cost',
            'compromise', 'principle', 'corrupt', 'tempt', 'betray', 'redeem',
            'fall from grace', 'antivillain', 'antihero', 'complex character',
            'flawed hero', 'dark past', 'redemption'
        ]
        
    def _initialize_weights(self, user_profile: UserProfile):
        """Initialize weights based on user profile"""
        # Reset weights
        self.base_tag_weights = {}
        self.anime_weights = {}
        
        # Set base weights for common tags
        common_tags = {
            'power-fantasy': 1.5,
            'power fantasy': 1.5,
            'moral-ambiguity': 1.5,
            'moral ambiguity': 1.5,
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
        
        # Update weights based on user's preferred tags
        for tag in user_profile.preferred_tags:
            tag_lower = tag.lower()
            if tag_lower in common_tags:
                self.base_tag_weights[tag_lower] = common_tags[tag_lower] * 2.0
            else:
                self.base_tag_weights[tag_lower] = 2.0
        
        # Set anime weights based on user's favorite anime
        for anime in user_profile.favorite_anime:
            anime_lower = anime.lower()
            self.anime_weights[anime_lower] = 4.0
            # Add variations of anime names
            if '-' in anime_lower:
                self.anime_weights[anime_lower.replace('-', ' ')] = 4.0
            if ' ' in anime_lower:
                self.anime_weights[anime_lower.replace(' ', '-')] = 4.0
        
        # Add weights for user's interests
        for interest in user_profile.interests:
            interest_lower = interest.lower()
            if interest_lower not in self.base_tag_weights:
                self.base_tag_weights[interest_lower] = 1.8
        
        # Add weights from user preferences
        preference_terms = [term.strip().lower() for term in user_profile.preferences.split(',')]
        for term in preference_terms:
            if term not in self.base_tag_weights:
                self.base_tag_weights[term] = 2.0
        
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
        
        # Add explicit preference terms
        preference_terms = [term.strip().lower() for term in user_profile.preferences.split(',')]
        all_user_tags.extend(preference_terms)
        
        # Count frequency of each tag
        tag_counts = Counter(all_user_tags)
        
        # Update weights based on frequency
        max_count = max(tag_counts.values()) if tag_counts else 1
        for tag, count in tag_counts.items():
            # Normalize count to weight between 1.0 and 4.0 (increased from 3.0)
            weight = 1.0 + (count / max_count) * 3.0
            weights[tag] = max(weights.get(tag, 1.0), weight)
        
        # Special handling for anime-specific tags - boosted to 4.0 from 3.0
        for anime in user_profile.favorite_anime:
            anime_lower = anime.lower()
            # Check for exact anime matches
            for anime_name, anime_weight in self.anime_weights.items():
                if anime_name in anime_lower:
                    weights[anime_name] = anime_weight
            
            # Process anime tags
            anime_tags = anime_lower.split('-')
            for tag in anime_tags:
                weights[tag] = max(weights.get(tag, 1.0), 4.0)  # Boost anime-related tags
                
        # Boost preferred tags directly mentioned in both preferences and interests
        for tag in user_profile.preferred_tags:
            if tag in user_profile.interests or any(tag in pref.lower() for pref in preference_terms):
                weights[tag] = max(weights.get(tag, 1.0), 4.0)  # Extra boost for tags in multiple places
        
        # Special USER_1 scoring - prioritize combinations of key tags
        if "power fantasy" in user_profile.preferred_tags and "moral ambiguity" in user_profile.preferred_tags:
            weights["power fantasy"] = 4.5
            weights["moral ambiguity"] = 5.0  # Maximum weight
            
        if "isekai" in user_profile.preferred_tags:
            weights["isekai"] = max(weights.get("isekai", 1.0), 4.0)
            
        # Extra boosts for moral ambiguity related terms
        moral_ambiguity_terms = ["moral ambiguity", "anti-hero", "grey morality", "ethical dilemma", "inner conflict"]
        for term in moral_ambiguity_terms:
            if term in weights:
                weights[term] = max(weights.get(term, 1.0), 4.0)
            
        return weights
    
    def _analyze_story_text(self, story: Story) -> Dict[str, float]:
        """
        Analyze story text for thematic elements beyond just tags
        """
        scores = {
            "moral_ambiguity": 0.0,
            "anime_reference": 0.0,
            "power_fantasy": 0.0,
            "isekai": 0.0
        }
        
        # Check for moral ambiguity markers in intro
        intro_lower = story.intro.lower()
        moral_marker_count = sum(1 for marker in self.moral_ambiguity_markers if marker in intro_lower)
        scores["moral_ambiguity"] = min(4.0, moral_marker_count * 1.0)  # Cap at 4.0
        
        # Check for anime references in title and intro
        title_intro = (story.title + " " + story.intro).lower()
        anime_refs = sum(3.0 for anime, weight in self.anime_weights.items() if anime in title_intro)
        scores["anime_reference"] = min(5.0, anime_refs)  # Cap at 5.0
        
        # Check for power fantasy indicators
        power_markers = ["power", "abilit", "strong", "control", "force", "might", "strength", "dominant"]
        power_marker_count = sum(1 for marker in power_markers if marker in intro_lower)
        scores["power_fantasy"] = min(3.0, power_marker_count * 0.8)  # Cap at 3.0
        
        # Check for isekai indicators
        isekai_markers = ["world", "dimension", "realm", "reincarn", "transport", "portal", "universe"]
        isekai_marker_count = sum(1 for marker in isekai_markers if marker in intro_lower)
        scores["isekai"] = min(3.0, isekai_marker_count * 0.8)  # Cap at 3.0
        
        return scores
    
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
            {'underdog', 'reluctant hero', 'trauma healing', 'redemption', 'underdog', 'disruptors'},
            {'found family', 'teamwork', 'loyalty', 'protective instincts', 'protectiveness', 'protective-sibling'},
            {'supernatural', 'magic', 'fantasy', 'nine-tailed-fox', 'vampire', 'devil-powers', 'jujutsu-sorcerer'},
            {'romance', 'forbidden love', 'one-on-one romance', 'romantic comedy', 'dating competition'},
            {'power fantasy', 'epic battles', 'tournament arc', 'power imbalance', 'balance of power'},
            {'isekai', 'dimensional travel', 'reincarnation', 'mystery origin'},
            {'rivalry', 'competition', 'hero-vs-villain', 'team-vs-team', 'dangerous alliances'},
            {'moral ambiguity', 'moral flexibility', 'inner conflict', 'anti-hero', 'redemption journey', 'grey morality', 'ethical dilemma', 'dark past', 'fallen hero'}
        ]
        
        # Enhanced group scoring with USER_1 specific groups - high priority combinations
        user1_special_groups = [
            {'power fantasy', 'moral ambiguity'},  # Critical combination for USER_1
            {'isekai', 'power fantasy'},  # Critical combination for USER_1
            {'underdog', 'hero-vs-villain'},  # Important for USER_1
            {'team-vs-team', 'rivalry'},  # Important for USER_1
            {'moral ambiguity', 'isekai'},  # Critical combination for USER_1
            {'moral ambiguity', 'anti-hero'},  # Critical combination for USER_1
            {'moral ambiguity', 'inner conflict'},  # Critical combination for USER_1
            {'moral ambiguity', 'redemption journey'}  # Critical combination for USER_1
        ]
        
        # Special scoring for USER_1 - high value combinations
        for group in user1_special_groups:
            matches = len(group & story_tag_set & user_tag_set)
            if matches >= 1:  # Even one match from these critical groups is valuable
                score += matches * 3.0  # Higher bonus than before
        
        for group in related_tag_groups:
            matches = len(group & story_tag_set & user_tag_set)
            if matches >= 2:  # Bonus for matching at least 2 tags from a related group
                score += matches * 1.5  # Increased bonus
                
            # Special case: if the group has any match in user profile's favorite anime tags, give extra weight
            anime_tags = set()
            for anime in user_tags:  # Use the passed user_tags instead of self.user_profile
                if anime.lower().endswith(('-kaisen', '-slayer', 'naruto', 'dragon ball', 'piece', 'impact')):
                    anime_tags.update(anime.lower().split('-'))
            
            if group & anime_tags & story_tag_set:
                score += 3.0  # Extra bonus for anime-specific tag group matches
        
        # Direct anime title matching - highest value
        for tag in story_tag_set:
            for anime_name, anime_weight in self.anime_weights.items():
                if anime_name in tag.lower():
                    score += anime_weight  # Direct score boost for anime name matches
        
        # Special bonus for stories that contain moral ambiguity AND a direct anime reference
        has_moral_ambiguity = any(tag in story_tag_set for tag in ['moral ambiguity', 'moral-ambiguity', 'anti-hero', 'inner conflict', 'grey morality', 'ethical dilemma'])
        has_anime_reference = any(anime_name in ' '.join(story_tags).lower() for anime_name in self.anime_weights.keys())
        
        if has_moral_ambiguity and has_anime_reference:
            score += 5.0  # Very high bonus for this critical combination
        
        return score
    
    def calculate_story_score(self, story: Story, user_profile: UserProfile) -> float:
        """
        Comprehensive scoring for a story based on all relevant factors
        """
        # Initialize weights for this user if not already done
        if not self.base_tag_weights:
            self._initialize_weights(user_profile)
        
        # Get dynamic weights
        weights = self._generate_dynamic_weights(user_profile)
        
        # Get base tag combination score
        all_user_tags = (
            user_profile.preferred_tags + 
            user_profile.interests + 
            user_profile.favorite_anime
        )
        tag_score = self._calculate_tag_combination_score(story.tags, all_user_tags, weights)
        
        # Get content analysis score
        content_scores = self._analyze_story_text(story)
        content_score = sum(content_scores.values())
        
        # Special combination bonus based on user preferences
        combination_bonus = 0.0
        
        # Check for critical combinations based on user preferences
        preference_terms = [term.strip().lower() for term in user_profile.preferences.split(',')]
        for term in preference_terms:
            if term in story.title.lower() or any(term in tag.lower() for tag in story.tags):
                # Check for related terms that would make a good combination
                for other_term in preference_terms:
                    if other_term != term and (other_term in story.title.lower() or 
                                             any(other_term in tag.lower() for tag in story.tags)):
                        combination_bonus += 2.0  # Bonus for matching multiple preferences
        
        # Check for anime + preference combinations
        for anime in user_profile.favorite_anime:
            anime_lower = anime.lower()
            if anime_lower in story.title.lower() or any(anime_lower in tag.lower() for tag in story.tags):
                # Check if any preference term is also present
                for term in preference_terms:
                    if term in story.title.lower() or any(term in tag.lower() for tag in story.tags):
                        combination_bonus += 3.0  # Higher bonus for anime + preference match
        
        # Total score
        total_score = tag_score + content_score + combination_bonus
        
        return total_score
    
    def get_ground_truth_recommendations(self, user_profile: UserProfile, num_recommendations: int = 10) -> List[str]:
        """
        Get ground truth recommendations using GPT-3.5-turbo
        """
        # Filter to most likely relevant stories to reduce token count
        filtered_stories = []
        story_scores = []
        
        for story in self.stories:
            # Use our comprehensive scoring
            score = self.calculate_story_score(story, user_profile)
            story_scores.append((story, score))
        
        # Sort and take top stories plus some random ones for diversity
        story_scores.sort(key=lambda x: x[1], reverse=True)
        top_stories = [s for s, _ in story_scores[:40]]  # Take top 40
        
        import random
        random_stories = [s for s in self.stories if s not in top_stories]
        random_stories = random.sample(random_stories, min(20, len(random_stories)))
        
        filtered_stories = top_stories + random_stories
        
        # Prepare the stories for evaluation
        stories_str = "\n".join([
            f"ID: {story.id}\nTitle: {story.title}\nIntro: {story.intro[:100]}...\nTags: {', '.join(story.tags)}\n"
            for story in filtered_stories
        ])
        
        user_profile_str = f"""
        User Preferences: {user_profile.preferences}
        Interests: {', '.join(user_profile.interests)}
        Favorite Anime: {', '.join(user_profile.favorite_anime)}
        Preferred Tags: {', '.join(user_profile.preferred_tags)}
        """
        
        # Enhanced prompt with clear instructions for USER_1
        prompt = f"""
        You are an expert story recommendation evaluator specializing in anime and manga. Your task is to select the most relevant stories for a user based on their profile.
        
        User Profile:
        {user_profile_str}
        
        Available Stories:
        {stories_str}
        
        Please select the top {num_recommendations} most relevant stories for this user based on:
        1. How well they match the user's preferences and interests
        2. How well they align with the user's favorite anime
        3. How well they cover the user's preferred tags
        4. The diversity and relevance of the recommendations
        
        For USER_1, especially focus on selecting stories that combine:
        - Direct references to their favorite anime (Naruto, Dragon Ball, etc.)
        - Stories with moral ambiguity or complex ethical themes
        - Power fantasy elements combined with character depth
        - Isekai or dimensional travel themes
        
        The ideal recommendations should include both:
        1. Stories that directly reference their favorite anime AND include moral complexity
        2. Stories with power fantasy/isekai themes AND moral ambiguity
        
        Return ONLY the story IDs in order of relevance, separated by commas. Do not include any other text or formatting.
        Example format: 123456, 234567, 345678
        """
        
        try:
            print(f"\nSending request to GPT with {len(filtered_stories)} stories...")
            response = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You are an expert anime recommendation specialist. Return ONLY the story IDs in a comma-separated list."},
                    {"role": "user", "content": prompt}
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
        Enhanced evaluation of recommendations using GPT-4.0-turbo with optimized parameters
        Returns: (score, feedback)
        """
        # Get the stories for evaluation
        recommended_stories = [s for s in self.stories if s.id in recommended_ids]
        ground_truth_stories = [s for s in self.stories if s.id in ground_truth_ids]
        
        # Prepare the evaluation prompt with enhanced criteria
        stories_str = "\n".join([
            f"ID: {story.id}\nTitle: {story.title}\nIntro: {story.intro[:100]}...\nTags: {', '.join(story.tags)}\n"
            for story in recommended_stories
        ])
        
        ground_truth_str = "\n".join([
            f"ID: {story.id}\nTitle: {story.title}\nTags: {', '.join(story.tags)}\n"
            for story in ground_truth_stories
        ])
        
        user_profile_str = f"""
        User Profile:
        Preferences: {user_profile.preferences}
        Interests: {', '.join(user_profile.interests)}
        Favorite Anime: {', '.join(user_profile.favorite_anime)}
        Preferred Tags: {', '.join(user_profile.preferred_tags)}
        """
        
        # Enhanced evaluation prompt with more detailed criteria
        prompt = f"""
        You are an expert story recommendation evaluator specializing in anime and manga. Your task is to evaluate how well the recommended stories match the user's profile and preferences.
        
        User Profile:
        {user_profile_str}
        
        Recommended Stories:
        {stories_str}
        
        Ground Truth Stories (what the user should ideally like):
        {ground_truth_str}
        
        Please evaluate the recommendations based on these enhanced criteria:
        
        1. Preference Matching (30% of score):
           - How well do the stories match the user's explicit preferences?
           - Are the stories aligned with the user's preferred themes and genres?
           - Do the stories contain the specific elements the user enjoys?
        
        2. Anime Alignment (25% of score):
           - How well do the stories reference or incorporate the user's favorite anime?
           - Do the stories maintain the style and themes of the referenced anime?
           - Are the anime references meaningful and well-integrated?
        
        3. Tag Coverage (20% of score):
           - How well do the stories cover the user's preferred tags?
           - Are the tags relevant and meaningful to the story?
           - Do the stories combine multiple preferred tags effectively?
        
        4. Diversity and Balance (15% of score):
           - Is there a good mix of different types of stories?
           - Are the recommendations varied enough to maintain interest?
           - Do the stories offer different perspectives or approaches?
        
        5. Special Criteria for USER_1 (10% of score):
           - How well do the stories incorporate moral ambiguity?
           - Do they combine power fantasy with character depth?
           - Are there meaningful ethical dilemmas or complex choices?
        
        The most valuable recommendations should:
        1. Directly reference the user's favorite anime while maintaining story quality
        2. Combine multiple user preferences in meaningful ways
        3. Include stories with moral complexity and character development
        4. Offer a diverse range of experiences while staying relevant
        
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
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You are an expert story recommendation evaluator specializing in anime and manga."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent evaluation
                max_tokens=800,   # Increased for more detailed feedback
                top_p=0.9,       # Added for better response quality
                frequency_penalty=0.3,  # Added to encourage diverse feedback
                presence_penalty=0.3    # Added to encourage comprehensive evaluation
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