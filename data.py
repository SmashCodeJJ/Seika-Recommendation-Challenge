from dataclasses import dataclass
from typing import List, Dict
import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class Story:
    id: str
    title: str
    intro: str
    tags: List[str]

@dataclass
class UserProfile:
    preferences: str
    interests: List[str]
    favorite_anime: List[str]
    preferred_tags: List[str]

def generate_more_stories(num_stories: int = 95) -> List[Story]:
    """
    Generate more sample stories using GPT-3.5-turbo
    Makes multiple API calls if needed to generate the requested number of stories
    """
    stories = []
    stories_per_batch = 15  # Reduced batch size
    remaining_stories = num_stories
    
    while remaining_stories > 0:
        current_batch = min(stories_per_batch, remaining_stories)
        prompt = f"""
        Generate {current_batch} new story entries that cater to three different user profiles. Each story should have:
        - A unique ID (6 digits, different from previous stories)
        - A creative title
        - An engaging intro
        - Relevant tags (5-7 tags per story)
        
        The stories should cover these themes and preferences:

        USER 1 Themes:
        - Power fantasy, moral ambiguity, isekai escapism
        - Underdog, rivalry, team-vs-team, hero-vs-villain
        - Master-servant, royalty-commoner, captor-captive dynamics
        - Romance, forbidden-love, love-triangles, found-family
        - Enemies-to-lovers, slow-burn
        - Reincarnation, devil-powers, jujutsu-sorcerer
        - Betrayal, loyalty, survival, redemption
        - Anime: Naruto, Dragon Ball, Jujutsu-Kaisen, Genshin-Impact, One-Piece, Demon-Slayer, Chainsaw-Man, Marvel/DC

        USER 2 Themes:
        - Reluctant/supportive guardian, disguised royalty, rookie competitor
        - Cafes, academies, fantasy kingdoms (Konoha, Hogwarts, Teyvat), cities
        - Supernatural/contemporary/historical romance
        - Supernatural beings, magic/curses/quirks
        - Harem, love triangles, power imbalance
        - Enemies-to-lovers, underdog, redemption
        - Forbidden desires, rival advances, legacy
        - Anime: Re:Zero, Naruto, My Hero Academia

        USER 3 Themes:
        - Underdog, reluctant hero, dominant protector
        - One-on-one romance, found-family bonds
        - Intense angst, trauma healing
        - Supernatural (nine-tailed foxes, vampires, magic)
        - Achievement-hunting, epic conclusions
        - Morally flexible exploration
        - Leaderboard climbing, protective sibling loyalty, guilt

        Sample format:
        ID: 217107
        Title: Stranger Who Fell From The Sky
        Intro: You are Devin, plummeting towards Orario with no memory of how you got here...
        Tags: danmachi, reincarnation, heroic aspirations, mystery origin, teamwork, loyalty, protectiveness

        Generate {current_batch} new stories in this exact format, ensuring a good mix of themes from all three users.
        Make sure to include:
        - Action-packed tournament arcs
        - Academy/school life stories
        - Supernatural romance
        - Fantasy kingdom adventures
        - Cafe slice-of-life with supernatural elements
        - Power fantasy isekai
        - Found family dynamics
        - Redemption arcs
        - Rivalry stories
        - Forbidden love scenarios
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative story generator for an anime-style interactive fiction platform."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=3500  # Adjusted token limit
            )
            
            stories_text = response.choices[0].message.content.strip()
            
            # Parse the generated stories
            current_id = None
            current_title = None
            current_intro = None
            current_tags = []
            batch_stories = []
            
            for line in stories_text.split('\n'):
                line = line.strip()
                if line.startswith('ID:'):
                    if current_id is not None:
                        batch_stories.append(Story(
                            id=current_id,
                            title=current_title,
                            intro=current_intro,
                            tags=current_tags
                        ))
                    current_id = line.split(':')[1].strip()
                    current_title = None
                    current_intro = None
                    current_tags = []
                elif line.startswith('Title:'):
                    current_title = line.split(':')[1].strip()
                elif line.startswith('Intro:'):
                    current_intro = line.split(':')[1].strip()
                elif line.startswith('Tags:'):
                    current_tags = [tag.strip() for tag in line.split(':')[1].split(',')]
            
            # Add the last story
            if current_id is not None:
                batch_stories.append(Story(
                    id=current_id,
                    title=current_title,
                    intro=current_intro,
                    tags=current_tags
                ))
            
            # Add valid stories to the main list
            for story in batch_stories:
                if story.id and story.title and story.intro and story.tags:
                    stories.append(story)
            
            # Update remaining stories count
            remaining_stories = num_stories - len(stories)
            print(f"Generated {len(stories)} stories so far...")
            
            if len(stories) >= num_stories:
                break
                
        except Exception as e:
            print(f"Error in story generation: {str(e)}")
            continue
    
    # Ensure we have exactly the requested number of stories
    return stories[:num_stories]

# Original sample stories
SAMPLE_STORIES = [
    Story(
        id="217107",
        title="Stranger Who Fell From The Sky",
        intro="You are Devin, plummeting towards Orario with no memory of how you got here...",
        tags=["danmachi", "reincarnation", "heroic aspirations", "mystery origin", "teamwork", "loyalty", "protectiveness"]
    ),
    Story(
        id="273613",
        title="Trapped Between Four Anime Legends!",
        intro="You're caught in a dimensional rift with four anime icons. Goku wants to spar...",
        tags=["crossover", "jujutsu kaisen", "dragon ball", "naruto", "isekai", "dimensional travel", "reverse harem"]
    ),
    Story(
        id="235701",
        title="New Transfer Students vs. Class 1-A Bully",
        intro="You and Zeroku watch in disgust as Bakugo torments Izuku again...",
        tags=["my hero academia", "challenging authority", "bullying", "underdog", "disruptors"]
    ),
    Story(
        id="214527",
        title="Zenitsu Touched Your Sister's WHAT?!",
        intro="Your peaceful afternoon at the Butterfly Estate shatters when Zenitsu accidentally gropes Nezuko...",
        tags=["demon slayer", "protective instincts", "comedic panic", "violent reactions"]
    ),
    Story(
        id="263242",
        title="Principal's Daughter Dating Contest",
        intro="You are Yuji Itadori, facing off against Tanjiro and Naruto for Ochako's heart...",
        tags=["crossover", "romantic comedy", "forced proximity", "harem", "dating competition"]
    )
]

def save_stories_to_file(stories: List[Story], filename: str = "stories.json"):
    """
    Save stories to a JSON file
    """
    stories_data = [
        {
            "id": story.id,
            "title": story.title,
            "intro": story.intro,
            "tags": story.tags
        }
        for story in stories
    ]
    
    with open(filename, 'w') as f:
        json.dump(stories_data, f, indent=2)
    print(f"Saved {len(stories)} stories to {filename}")

def load_stories_from_file(filename: str = "stories.json") -> List[Story]:
    """
    Load stories from a JSON file
    """
    try:
        with open(filename, 'r') as f:
            stories_data = json.load(f)
        
        stories = [
            Story(
                id=story["id"],
                title=story["title"],
                intro=story["intro"],
                tags=story["tags"]
            )
            for story in stories_data
        ]
        print(f"Loaded {len(stories)} stories from {filename}")
        return stories
    except FileNotFoundError:
        print(f"File {filename} not found, will generate new stories")
        return None

def get_stories() -> List[Story]:
    """
    Get all available stories, either from file or by generating new ones
    """
    # Try to load stories from file first
    stories = load_stories_from_file()
    
    if stories is None:
        # If no file exists, generate new stories
        stories = SAMPLE_STORIES + generate_more_stories()
        # Save the generated stories for future use
        save_stories_to_file(stories)
    
    return stories

# Sample users
SAMPLE_USERS = {
    "USER_1": UserProfile(
        preferences="Power fantasy, moral ambiguity, isekai escapism",
        interests=["underdog", "rivalry", "team-vs-team", "hero-vs-villain"],
        favorite_anime=["Naruto", "Dragon Ball", "Jujutsu-Kaisen", "Genshin-Impact", "One-Piece", "Demon-Slayer", "Chainsaw-Man", "Marvel/DC"],
        preferred_tags=["power fantasy", "moral ambiguity", "isekai", "underdog", "rivalry", "team-vs-team", "hero-vs-villain"]
    ),
    "USER_2": UserProfile(
        preferences="Reluctant/supportive guardian, disguised royalty, rookie competitor",
        interests=["cafes", "academies", "fantasy kingdoms", "supernatural romance"],
        favorite_anime=["Re:Zero", "Naruto", "My Hero Academia"],
        preferred_tags=["reluctant guardian", "academy", "supernatural", "romance", "power imbalance"]
    ),
    "USER_3": UserProfile(
        preferences="Underdog, reluctant hero, dominant protector",
        interests=["one-on-one romance", "found-family bonds", "intense angst", "trauma healing"],
        favorite_anime=["Naruto", "Demon Slayer", "Jujutsu Kaisen"],
        preferred_tags=["underdog", "reluctant hero", "supernatural", "found family", "trauma healing"]
    )
}

# For backward compatibility
SAMPLE_USER = SAMPLE_USERS["USER_1"]

USER_2 = UserProfile(
    preferences="Self-insert choice-driven narrator as reluctant/supportive guardian, disguised royalty, rookie competitor",
    interests=["cafes", "academies", "fantasy kingdoms", "cities"],
    favorite_anime=["Re:Zero", "Naruto", "My Hero Academia"],
    preferred_tags=[
        "supernatural", "contemporary", "historical romance", "fantasy", "action", "horror",
        "harem", "love-triangle", "power-imbalance", "enemies-to-lovers", "underdog", "redemption",
        "forbidden-desires", "rivalry", "legacy"
    ]
)

USER_3 = UserProfile(
    preferences="Male roleplayer seeking immersive, choice-driven narratives",
    interests=[
        "self-insert underdog", "reluctant hero", "dominant protector",
        "one-on-one romance", "found-family bonds", "intense angst", "trauma healing"
    ],
    favorite_anime=["Naruto", "Demon Slayer", "Jujutsu Kaisen"],
    preferred_tags=[
        "supernatural", "nine-tailed-fox", "vampire", "magic",
        "achievement-hunting", "epic-conclusions", "morally-flexible",
        "leaderboard", "protective-sibling", "loyalty", "guilt"
    ]
) 