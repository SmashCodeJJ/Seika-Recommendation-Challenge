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
    """
    prompt = f"""
    Generate {num_stories} new story entries in the same format as the sample stories below.
    Each story should have:
    - A unique ID (6 digits)
    - A creative title
    - An engaging intro
    - Relevant tags (5-7 tags per story)
    
    Sample format:
    ID: 217107
    Title: Stranger Who Fell From The Sky
    Intro: You are Devin, plummeting towards Orario with no memory of how you got here...
    Tags: danmachi, reincarnation, heroic aspirations, mystery origin, teamwork, loyalty, protectiveness
    
    Generate {num_stories} new stories in this exact format, one after another.
    Make sure the stories are diverse in themes and genres, including:
    - Isekai adventures
    - School life
    - Fantasy battles
    - Romance
    - Mystery
    - Action
    - Comedy
    - Drama
    - Supernatural
    - Sci-fi
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative story generator for an anime-style interactive fiction platform."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=4000
    )
    
    stories_text = response.choices[0].message.content.strip()
    stories = []
    
    # Parse the generated stories
    current_id = None
    current_title = None
    current_intro = None
    current_tags = []
    
    for line in stories_text.split('\n'):
        line = line.strip()
        if line.startswith('ID:'):
            if current_id is not None:
                stories.append(Story(
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
        stories.append(Story(
            id=current_id,
            title=current_title,
            intro=current_intro,
            tags=current_tags
        ))
    
    return stories

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

# Generate more stories
EXPANDED_STORIES = SAMPLE_STORIES + generate_more_stories()

def load_stories() -> List[Story]:
    """Load stories from the stories.json file"""
    try:
        with open('stories.json', 'r') as f:
            stories_data = json.load(f)
            return [Story(**story) for story in stories_data]
    except FileNotFoundError:
        print("Warning: stories.json not found. Using sample stories.")
        return EXPANDED_STORIES

# Sample user profile
SAMPLE_USER = UserProfile(
    preferences="I enjoy action-packed stories with strong character development and interesting plot twists.",
    interests=["martial arts", "supernatural powers", "school life"],
    favorite_anime=["Naruto", "My Hero Academia", "One Piece"],
    preferred_tags=["action", "power-fantasy", "isekai", "crossover"]
) 