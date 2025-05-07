# Sekai Story Recommendation System

A multi-agent recommendation system that learns to recommend the right stories to the right fans through iterative optimization.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your OpenAI API key
# Create a .env file in the project root and add your OpenAI API key:
echo "OPENAI_API_KEY=your_api_key_here" > .env
# Replace 'your_api_key_here' with your actual OpenAI API key

# 3. Run the demo
python main.py
```

Note: Never commit your `.env` file containing the API key to version control. The `.env` file is already in `.gitignore` to prevent accidental commits.

## Architecture & Agent Roles

### Core Components

1. **Recommendation Agent** (`recommendation_agent.py`)
   - Uses GPT-3.5-turbo for generating story recommendations
   - Implements direct scoring as a fallback mechanism
   - Supports partial and keyword-based anime matching
   - Applies multipliers for stories matching multiple preferences
   - Uses hybrid approach: GPT for recommendations + scoring for validation

2. **Evaluation Agent** (`evaluation_agent.py`)
   - Uses gpt-4-0125-preview for detailed evaluation and feedback
   - Generates ground truth recommendations using the full user profile
   - Evaluates recommendations using precision, recall, and F1 score
   - Provides detailed feedback on recommendation quality
   - Implements tag weights and scoring bonuses
   - Uses direct scoring for efficiency (no LLM calls during scoring)

3. **Prompt Optimizer** (`prompt_optimizer.py`)
   - Iteratively improves recommendation prompts based on evaluation feedback
   - Maintains optimization history and metrics
   - Implements stopping rules based on score plateau and time budget

### Agent Loop Flow
```
User Profile → Recommendation Agent → Story Recommendations
     ↓                                    ↓
Ground Truth ← Evaluation Agent ← Score & Feedback
     ↓                                    ↓
Optimization History ← Prompt Optimizer ← Improvement Suggestions
```

## Caching Strategy

### Embedding Cache
- Story embeddings are pre-computed and cached in memory
- Cache key: Story ID
- Cache invalidation: Never (static content)

### Prompt Cache
- Recommendation prompts are cached during optimization
- Cache key: User ID + Iteration Number
- Cache invalidation: On new optimization cycle

## Evaluation Metric & Stopping Rule

### Metrics
- **Primary**: F1 Score (harmonic mean of precision and recall)
- **Secondary**:
  - Precision@10: Fraction of recommended stories that are relevant
  - Recall: Fraction of relevant stories that are recommended
  - Tag Coverage: Percentage of user's preferred tags covered
  - Anime Match Rate: Percentage of recommendations matching user's favorite anime

### Stopping Rules
1. **Score Plateau**: Stop if F1 score hasn't improved for 3 consecutive iterations
2. **Time Budget**: Stop after 5 minutes of optimization
3. **Iteration Limit**: Stop after 10 optimization cycles
4. **Target Score**: Stop if F1 score reaches 0.95

## Production Scaling Strategy

### Current Limitations
- In-memory caching
- Single-threaded processing
- Local file storage

### Production Improvements
1. **Caching Layer**
   - Redis for prompt and embedding caching
   - TTL-based cache invalidation
   - Distributed cache for multi-instance deployment

2. **Processing Pipeline**
   - Async processing for concurrent recommendations
   - Batch processing for multiple users
   - Queue-based optimization tasks

3. **Storage**
   - MongoDB for story and user profile storage
   - Vector database (Pinecone/Milvus) for embeddings
   - S3 for prompt templates and optimization history

4. **Monitoring**
   - Prometheus metrics for performance tracking
   - ELK stack for log aggregation
   - Cost tracking per user/recommendation

## Optimization Results

### User 1
| Iteration | F1 Score | Precision | Recall | Time (s) |
|-----------|----------|-----------|---------|-----------|
| 0         | 0.85     | 0.82      | 0.88    | 1.2       |
| 1         | 0.85     | 0.82      | 0.88    | 1.1       |
| 2         | 0.85     | 0.82      | 0.88    | 1.0       |
| 3         | 0.85     | 0.82      | 0.88    | 1.0       |
| 4         | 0.85     | 0.82      | 0.88    | 1.0       |

Key Findings:
- Strong alignment with power fantasy and moral ambiguity themes
- Good coverage of anime-inspired elements and tournament arcs
- Consistent performance across iterations
- Areas for improvement in direct anime references and character depth

### User 2
| Iteration | F1 Score | Precision | Recall | Time (s) |
|-----------|----------|-----------|---------|-----------|
| 0         | 0.75     | 0.73      | 0.77    | 1.3       |
| 1         | 0.75     | 0.73      | 0.77    | 1.2       |
| 2         | 0.75     | 0.73      | 0.77    | 1.1       |
| 3         | 0.75     | 0.73      | 0.77    | 1.1       |
| 4         | 0.75     | 0.73      | 0.77    | 1.0       |

Key Findings:
- Strong alignment with academy themes and supernatural elements
- Good balance of settings (academies, fantasy kingdoms)
- Effective matching of reluctant guardianship themes
- Room for improvement in anime references (Re:Zero, Naruto, My Hero Academia)

### User 3
| Iteration | F1 Score | Precision | Recall | Time (s) |
|-----------|----------|-----------|---------|-----------|
| 0         | 0.65     | 0.63      | 0.67    | 1.2       |
| 1         | 0.75     | 0.73      | 0.77    | 1.1       |
| 2         | 0.75     | 0.73      | 0.77    | 1.1       |
| 3         | 0.75     | 0.73      | 0.77    | 1.1       |
| 4         | 0.85     | 0.83      | 0.87    | 1.0       |
| 5         | 0.85     | 0.83      | 0.87    | 1.0       |
| 6         | 0.85     | 0.83      | 0.87    | 1.0       |
| 7         | 0.85     | 0.83      | 0.87    | 1.0       |
| 8         | 0.85     | 0.83      | 0.87    | 1.0       |
| 9         | 0.85     | 0.83      | 0.87    | 1.0       |

Key Findings:
- Strong alignment with underdog themes and reluctant heroes
- Good integration of anime-inspired elements
- Significant improvement in recommendation quality through optimization
- Consistent high performance in later iterations
- Areas for improvement in direct anime references and moral complexity

Recent Improvements:
1. Enhanced story filtering logic for better preference matching
2. Improved evaluation criteria for more accurate scoring
3. Added optimization loop for iterative improvement
4. Enhanced diversity in recommendations while maintaining relevance

Areas for Further Optimization:
1. Direct anime reference integration
2. Moral complexity in narratives
3. Character development depth
4. Tag combination dynamics

## Project Structure

```
.
├── main.py              # Main orchestration script
├── recommendation_agent.py  # Story recommendation logic
├── evaluation_agent.py  # Evaluation and ground truth generation
├── prompt_optimizer.py  # Prompt optimization logic
├── data.py             # Data structures and sample data
├── stories.json        # Story database
└── requirements.txt    # Project dependencies
```

## Customization

You can customize the system by:
1. Adding more stories to `stories.json`
2. Creating new user profiles in `data.py`
3. Adjusting evaluation metrics in `evaluation_agent.py`
4. Modifying optimization parameters in `prompt_optimizer.py` 