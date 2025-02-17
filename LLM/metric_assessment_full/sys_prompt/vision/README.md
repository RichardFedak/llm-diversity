| Evaluation                                  | Total Evaluations | LLM/User Correlation (%) | LLM/CB-ILD Correlation (%) | LLM/CF-ILD Correlation (%) | LLM/BIN-DIV Correlation (%) | CB-ILD Chosen (%) | CF-ILD Chosen (%) | BIN-DIV Chosen (%) |
|---------------------------------------------|--------------------|------------------------|------------------------|------------------------|-------------------------|----------------------|----------------------|----------------------|
| [covers_summary](covers_summary.py)        | 107                | 40.19                  | 31.78                  | 18.69                  | **53.27**                   | 31.87                  | 16.82                  | **51.40**                  |
| [covers_think](covers_think.py)            | 107                | 40.19                  | **45.79**                  | 27.10                  | 35.51                   | **45.79**                  | 19.63                  | 34.58                  |
| [covers_think_titles](covers_think_titles.py) | 107                | 36.45                  | **40.19**                  | 30.84                  | 35.51                   | **40.19**                  | 24.30                  | 34.58                  |
| [covers_think_genres](covers_think_genres.py) | 107                | **43.93**                  | 24.30                  | 34.58                  | **49.53**                   | 24.30                  | 27.10                  | **48.60**                  |
| [covers_think_plot](covers_think_plot.json) | 107                | **43.93**                  | **42.99**                  | 22.43                  | **42.99**                   | **42.99**                  | 17.76                  | **42.99**                  |
| [covers_think_full](covers_think_full.py)   | 107                | 37.38                  | 34.58                  | 29.91                  | 39.25                   | 34.58                  | 27.10                  | 38.32                  |


**Findings:**

*   As noted in related [ILS and diversity paper](https://link.springer.com/article/10.1007/s11257-022-09351-w), the most frequent response regarding the criteria that determine movie similarity (and diversity) was genre and plot. These evaluations have highest correlation with user responses.

*   The `CB-ILD` *(cos. dist. of embeddings of plot)* metric shows the highest correlation with the LLM in the `covers_think`, `covers_think_titles` & `covers_think_plot`. In response summary, we see focus on genres, themes, visuals, but it correlates the most with the plot embedding diversity, interesting...

*   No correlation with `CF-ILD` *(cosine dist. of rating vectors of items)* metric at all.

*   `BIN-DIV` *(coverage x non-redundancy of genres)* correlates with `covers_summary` & `covers_think_genres`, makes sense.

*   The overall coverage, that is at least one correct output from one of the evaluations, is ***~89%***. `covers_think_genres.py` & `covers_summary.py` have coverage of ***~64%***. With `covers_think_plot` ***~75%***.