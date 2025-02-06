| Evaluation                                  | Total Evaluations | LLM/User Correlation (%) | LLM/CB-ILD Correlation (%) | LLM/CF-ILD Correlation (%) | LLM/BIN-DIV Correlation (%) | CB-ILD Chosen (%) | CF-ILD Chosen (%) | BIN-DIV Chosen (%) |
|---------------------------------------------|--------------------|------------------------|------------------------|------------------------|-------------------------|----------------------|----------------------|----------------------|
| [covers_summary](covers_summary.py)        | 107                | **42.06%**                | 30.84%                | 21.50%                | **54.21%**                 | 30.84%               | 16.82%               | **52.34%**              |
| [covers_think](covers_think.py)            | 107                | 27.10%                | 29.91%                | 38.32%                | 27.10%                 | 29.91%               | **41.12%**               | 28.97%               |
| [covers_think_full](covers_think_full.py)   | 107                | **44.86%**                | 34.58%                | 29.91%                | **42.06%**                 | 34.58%               | 24.30%               | **41.12%**               |
| [covers_think_genres](covers_think_genres.py) | 107                | **43.93%**                | 28.97%                | 30.84%                | **47.66%**                 | 28.97%               | 24.30%               | **46.73%**               |
| [covers_think_titles](covers_think_titles.py) | 107                | **44.86%**                | 38.32%                | 28.04%                | 39.25%                 | 38.32%               | 21.50%               | 40.19%               |

**Findings:**

*   As noted in related [ILS and diversity paper](https://link.springer.com/article/10.1007/s11257-022-09351-w), the most frequent response regarding the criteria that determine movie similarity (and diversity) was genre. Therefore, in the `covers_think` evaluation, where only movie covers were provided to the LLM, the LLM/User correlation is only 27%. Evaluations that included additional information about the movies resulted in higher LLM/User and LLM/BIN-DIV correlations. This suggests that visual information alone is not sufficient.

*   The CB-ILD metric shows the highest correlation with the LLM in the `covers_think` evaluation, where only movie covers were presented without supplementary information. So the **cosine distance of rating vectors** has **(a slight) correlation** with the **visual features** present on the movie covers ?
