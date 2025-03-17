# User Study

**Main goal**: Understand how users perceive diversity in movie recommendations and develop a system that personalizes recommendations based on their perception.

**Dataset**: Movielens

## Preference & diversity discovery

In this phase, the main goal is to understand the user, what type of movies they prefer, and how they perceive diversity.

### Preference elicitation

1.  **Movie Selection:** 
    -  Same as in the paper.?...
    -  Recommend random/popular + non-popular ***TODO***

### Diversity perception

Main focus is not to recommend movies suitable for user but to learn how they perceive diversity.

1.  **Diversity perception learning:** 
    -  Three lists of X(?) movies each. 
       -  ***TODO: Each iteration, a different number of movies in the list?***
       -  ***TODO: Lists generated in 3 different ways ? combination of popular/non-popular + CF,CB,BIN_DIV ILD ... OR focus on TOP 3 borda counts criteria (genres, plot, actors) from (https://link.springer.com/article/10.1007/s11257-022-09351-w/tables/12)***
    -  Select the list you perceive as most diverse (select multiple if you perceive their diversity as the same?).
    -  *(Ask the user why they selected that list, text-based/options [genres, plots ...])*
    -  Iterated Y(?) times.
2.  **Sample generation:**  
   - [LLM paper 3.1.2](https://arxiv.org/pdf/2306.05817)
   - LLM will generate lists ***OR*** generate lists like in the 1st part.
   - LLM selects the most diverse list(s) + reasoning?
   - Ask the user if they agree, why, why not...
   - Iterated Z(?) times.
3.  **Questionnaire?** --- ***TODO, after the learning phase?***  
   - Questions about the number of movies in lists...  
     - *How was it to determine the most diverse list when only a few movies were in each list?*  
     - *How was it to determine the most diverse list when there were many movies in each list?*  
     - *Which count of movies in lists suited you the best?*  
     - Give weights to selections / remove selections based on responses. (Keep only selections that user is confident with.)
   - Overall, was it easy to determine the more diverse lists, and are you happy with your selections? Yes/No - if ***NO*** - One more time (only with the selected movie count...) + discard all previous selections?  

## Recommendation

We should have some basic understanding of the user, their preferences, and how they perceive diversity.  
[LLM paper 3.3.2](https://arxiv.org/pdf/2306.05817)  
- **CRM Generation**
  - Not using LLM.
- **Open-set item generation**
  - Based on user preferences and the diversity learning phase, generate a list of movies.
  - Post-processing for generative hallucination. ***TODO*** Check if the movie exists in the dataset?
  - ***TODO*** Check possible ID-based generation... https://arxiv.org/pdf/2305.06569
- **Closed-set item generation**
  - Given also a candidate item set from CRM (pre-filter), shrink the set, and re-rank movies.
  - The number of candidates depends on the LLM context window...
- **Hybrid generation**
  - Different prompting templates for generating the candidate set and for re-ranking.

X lists of Y movies, each list using a different generation.  
X lists of Y movies, same generation, different parameters. 

**TASKS:**  
  - Select movies  
  - Rate the diversity of lists  
  - Make it possible for user to ask for more diverse list ? Something like "This list is not diverse, give me more diverse list..." + reasoning
  - ***TODO...***  

**Recommendations generation:**  
  - Based on evaluations for the paper [User Perception of Diversity](https://dl.acm.org/doi/pdf/10.1145/3627043.3659555), tuning a model did not prove to be an effective solution, as it produced similar or worse predictions than using system prompts and also took a lot more time to prepare...  
  - Approach in the experiment: **Not Tune LLM & Infer with/without CRM**  
  - Since LLM fails to extract useful info from long user behavior ([Diversity Perception Learning Phase](#diversity-perception-testing)), we need to focus on meaningful parts only. ***TODO***  
