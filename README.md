# Sekai Story Recommendation System

This is a multi-agent recommendation system that learns to recommend the right stories to the right fans. The system consists of three main agents:

1. **Recommendation Agent**: Uses GPT-3.5-turbo to recommend stories based on user profiles
2. **Evaluation Agent**: Evaluates recommendations against ground truth and provides feedback
3. **Prompt Optimizer**: Iteratively improves the recommendation prompts based on evaluation results

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the main script to start the recommendation system:
```bash
python main.py
```

The system will:
1. Generate ground truth recommendations
2. Iteratively optimize the recommendation prompts
3. Stop when the evaluation score plateaus or reaches the maximum number of iterations

## Project Structure

- `data.py`: Contains data structures and sample data
- `recommendation_agent.py`: Handles story recommendations
- `evaluation_agent.py`: Evaluates recommendations and provides feedback
- `prompt_optimizer.py`: Optimizes recommendation prompts
- `main.py`: Orchestrates the agents and runs the optimization loop

## Evaluation Metrics

The system uses:
- Precision@10: The fraction of recommended stories that are relevant
- Mean Recall: The fraction of relevant stories that are recommended
- F1 Score: The harmonic mean of precision and recall

## Customization

You can customize the system by:
1. Adding more stories to `SAMPLE_STORIES` in `data.py`
2. Creating new user profiles in `data.py`
3. Adjusting the evaluation metrics in `evaluation_agent.py`
4. Modifying the optimization criteria in `prompt_optimizer.py` 