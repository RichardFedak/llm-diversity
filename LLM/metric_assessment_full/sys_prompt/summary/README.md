| Evaluation                              | Accuracy | cf_ild   | cb_ild   | bin_div  |
|-----------------------------------------|----------|----------|----------|----------|
| [Titles](single_think.py)               | ***45.79%***   | 22.4%    | 29.0%    | ***56.1%***    |
| [T+Genres](single_think_genres.py)      | ***44.86%***   | 18.7%    | 28.0%    | ***59.8%***    |
| [T+G+Plot](single_think_full.py)        | ***42.99%***   | 18.7%    | 29.0%    | ***57.9%***    |
| [Standouts+Titles](single_popularity.py) | ***43.93%***   | 20.6%    | 27.1%    | ***57.9%***    |
| [Standouts+T+Genres](single_popularity_genres.py) | ***46.73%***   | 21.5%    | 33.6%    | ***55.1%***    |
| [Standouts+G+Plot](single_popularity_plot.py) | *39.25%*   | 19.6%    | 27.1%    | ***60.7%***    |

### Findings
- **Summary** increases correlation of MD-genres-BinDiv with Gemini accross all experiments dramatically

## Chosen metrics

| Evaluation                              | BIN-DIV  | CB-ILD   | CF-ILD   |
|-----------------------------------------|----------|----------|----------|
| [Titles](single_think.py)               | 54.21%   | 28.97%    | 15.89%    |
| [T+Genres](single_think_genres.py)      | 57.94%   | 28.04%    | 14.02%    |
| [T+G+Plot](single_think_full.py)        | 56.07%   | 28.97%    | 14.95%    |
| [Standouts+Titles](single_popularity.py) | 56.07%   | 27.10%    | 14.95%    |
| [Standouts+T+Genres](single_popularity_genres.py) | 53.27%   | 33.64%    | 13.08%    |
| [Standouts+G+Plot](single_popularity_plot.py) | 58.88%   | 27.10%    | 14.02%    |

