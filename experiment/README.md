# User Study

**Main goal**: Understand how users perceive diversity in movie recommendations and develop a system that personalizes recommendations based on their perception.

**Dataset**: Movielens

## Preference & diversity discovery

In this phase, the main goal is to understand the user, what type of movies they prefer, and how they perceive diversity.

1.  **Movie Selection:** 
    -  Same as in the paper.?...
    -  Recommend random/popular + non-popular ***TODO***

2.  **Diversity perception**
    -  One page
    -  Pairs of movies
       -  with different genres, plot, actory, release year
       -  from the same franchise
       -  same genres, different plot
       -  same actor/s, different genres
       -  same actor/s, same genres
       -  similar plot, different genres ? parody+original ?
    -  For each pair: 
       -  Are the movies different from each other ?
       -  Likert scale ?
       -  Write, what they have in common and what features are diverse

## Recommendation

We should have some basic understanding of the user, their preferences, and how they perceive diversity.  
[LLM paper 3.3.2](https://arxiv.org/pdf/2306.05817)  
- **CRM Generation**
  - Not using LLM. ***BIN-DIV***
- Vyskušat generovanie...
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
  - RELEVANCE
  - Make it possible for user to ask for more diverse list ? Something like "This list is not diverse, give me more diverse list..." + reasoning, textový feedback, conversational recsys PAPERS LLMs
  - napisat Patrikovi
  - ***TODO...***  

## Questionnaire
  - Which algo provided most diverse recommendations? Why...
  - Which algo provided least diverse recommendations? Why...

**Recommendations generation:**  
  - Based on evaluations for the paper [User Perception of Diversity](https://dl.acm.org/doi/pdf/10.1145/3627043.3659555), tuning a model did not prove to be an effective solution, as it produced similar or worse predictions than using system prompts and also took a lot more time to prepare...  
  - Approach in the experiment: **Not Tune LLM & Infer with/without CRM**  
  - Since LLM fails to extract useful info from long user behavior ([Diversity Perception Learning Phase](#diversity-perception-testing)), we need to focus on meaningful parts only. ***TODO***  
