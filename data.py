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
    
    # Track used IDs to ensure uniqueness
    used_ids = set()
    
    # First, collect all existing IDs from SAMPLE_STORIES
    for story in SAMPLE_STORIES:
        used_ids.add(story.id)
    
    while remaining_stories > 0:
        current_batch = min(stories_per_batch, remaining_stories)
        prompt = f"""
        Generate {current_batch} new story entries that cater to different user profiles. Each story should have:
        - A unique ID (6 digits, different from previous stories)
        - A creative title
        - An engaging intro
        - Relevant tags (5-7 tags per story)
        
        The stories should cover these themes and preferences:

        USER 4 Themes (Trickster Time-Looper):
        - Chaotic-good trickster mentor
        - Time-loop puzzle-solver
        - Cosmic-horror explorer
        - Unreliable-narrator twists
        - Fourth-wall breaks
        - Dark humour
        - Intellectual rivalry
        - Cat-and-mouse games
        - Countdown tension
        - Grudging camaraderie
        - Redemption arcs
        - References to: Steins;Gate, Higurashi, Loki (MCU), Control, Outer Wilds, Alan Wake

        Sample format:
        ID: 217107
        Title: Stranger Who Fell From The Sky
        Intro: You are Devin, plummeting towards Orario with no memory of how you got here...
        Tags: danmachi, reincarnation, heroic aspirations, mystery origin, teamwork, loyalty, protectiveness

        Generate {current_batch} new stories in this exact format, ensuring a good mix of themes.
        Make sure to include:
        - Time loop mysteries
        - Psychological horror elements
        - Cosmic horror scenarios
        - Unreliable narrator stories
        - Puzzle-box narratives
        - Trickster mentor dynamics
        - Fourth-wall breaking moments
        - Dark humor situations
        - Intellectual challenges
        - Redemption arcs
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
                        # Only add story if ID is unique
                        if current_id not in used_ids:
                            batch_stories.append(Story(
                                id=current_id,
                                title=current_title,
                                intro=current_intro,
                                tags=current_tags
                            ))
                            used_ids.add(current_id)
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
            
            # Add the last story if ID is unique
            if current_id is not None and current_id not in used_ids:
                batch_stories.append(Story(
                    id=current_id,
                    title=current_title,
                    intro=current_intro,
                    tags=current_tags
                ))
                used_ids.add(current_id)
            
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
        print(f"Error: File {filename} not found")
        return []

def get_stories() -> List[Story]:
    """
    Get all available stories from the stories.json file
    """
    stories = load_stories_from_file()
    if not stories:
        raise FileNotFoundError("stories.json file not found or empty")
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
    ),
    "USER_4": UserProfile(
        preferences="chaotic-good trickster mentor; time-loop puzzle-solver, cosmic-horror explorer",
        interests=[
            "unreliable-narrator twists", "fourth-wall breaks", "butterfly-effect stakes",
            "intellectual rivalry", "cat-and-mouse games", "countdown tension",
            "dark humour", "grudging camaraderie", "redemption arcs"
        ],
        favorite_anime=["Steins;Gate", "Higurashi", "Loki (MCU)", "Control", "Outer Wilds", "Alan Wake"],
        preferred_tags=[
            "timeloop", "psychological-horror", "cosmic-mystery", 
            "unreliable-narrator", "puzzle-box narrative"
        ]
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

def load_stories() -> List[Story]:
    """
    Load stories from file or generate new ones if file doesn't exist
    """
    try:
        return load_stories_from_file()
    except FileNotFoundError:
        stories = generate_more_stories()
        save_stories_to_file(stories)
        return stories

def load_user_profiles() -> Dict[str, UserProfile]:
    """
    Load user profiles from SAMPLE_USERS
    """
    return SAMPLE_USERS 

def parse_user_string(user_string: str) -> UserProfile:
    """
    Parse a user string in the format:
    "USER_X description; preferences; settings; genres; power-dynamics; emotional catalysts; fandom mix; wants; shuns; tags"
    """
    # Split the string into main components
    parts = user_string.split(';')
    
    # Extract user ID and preferences (first part)
    user_id, preferences = parts[0].split(' ', 1)
    
    # Initialize lists for different components
    interests = []
    favorite_anime = []
    preferred_tags = []
    
    # Process each part
    for part in parts[1:]:
        part = part.strip()
        if part.startswith('settings:'):
            interests.extend([s.strip() for s in part.replace('settings:', '').split(',')])
        elif part.startswith('genres:'):
            interests.extend([g.strip() for g in part.replace('genres:', '').split(',')])
        elif part.startswith('power-dynamics:'):
            interests.extend([p.strip() for p in part.replace('power-dynamics:', '').split(',')])
        elif part.startswith('emotional catalysts:'):
            interests.extend([e.strip() for e in part.replace('emotional catalysts:', '').split(',')])
        elif part.startswith('fandom mix:'):
            favorite_anime.extend([f.strip() for f in part.replace('fandom mix:', '').split(',')])
        elif part.startswith('wants:'):
            interests.extend([w.strip() for w in part.replace('wants:', '').split(',')])
        elif part.startswith('tags:'):
            preferred_tags.extend([t.strip() for t in part.replace('tags:', '').split(',')])
    
    # Create and return the UserProfile
    return UserProfile(
        preferences=preferences.strip(),
        interests=interests,
        favorite_anime=favorite_anime,
        preferred_tags=preferred_tags
    )

def add_user_to_sample_users(user_string: str) -> None:
    """
    Add a new user to SAMPLE_USERS dictionary from a formatted string
    """
    # Extract user ID from the string
    user_id = user_string.split(' ')[0]
    
    # Parse the user string into a UserProfile
    user_profile = parse_user_string(user_string)
    
    # Add to SAMPLE_USERS
    SAMPLE_USERS[user_id] = user_profile
    print(f"Added {user_id} to SAMPLE_USERS")

# Example usage:
# user_string = "USER_5 soft-spoken healer turned vengeance-seeker; prefers low-combat diplomacy routes that can snap to ruthless justice when bonds are broken; settings: post-apocalyptic wastelands, overgrown ruins, cosy survivor enclaves; genres: solarpunk hopepunk, found-family road-trip, creature-taming; power-dynamics: caretaker-ward, oath-breaker confrontation; emotional catalysts: betrayal of pacifist ideals, protecting children, rebuilding community; fandom mix: The Last of Us, Nausicaä, Atelier Ryza, The Walking Dead, Pokémon Legends: Arceus; wants crafting, herbalism, base-building; shuns nihilism and sexual violence; tags: healing-journey, hopepunk, creature-companion, settlement-sim, oath-betrayal"
# add_user_to_sample_users(user_string)

# Remove USER_5 setup
# user_string = "USER_5 soft-spoken healer turned vengeance-seeker; prefers low-combat diplomacy routes that can snap to ruthless justice when bonds are broken; settings: post-apocalyptic wastelands, overgrown ruins, cosy survivor enclaves; genres: solarpunk hopepunk, found-family road-trip, creature-taming; power-dynamics: caretaker-ward, oath-breaker confrontation; emotional catalysts: betrayal of pacifist ideals, protecting children, rebuilding community; fandom mix: The Last of Us, Nausicaä, Atelier Ryza, The Walking Dead, Pokémon Legends: Arceus; wants crafting, herbalism, base-building; shuns nihilism and sexual violence; tags: healing-journey, hopepunk, creature-companion, settlement-sim, oath-betrayal"
# add_user_to_sample_users(user_string) 