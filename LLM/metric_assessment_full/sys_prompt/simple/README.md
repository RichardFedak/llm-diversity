| Evaluation             | Accuracy | cf_ild  | cb_ild  | bin_div  |
|------------------------|----------|--------|--------|---------|
| [Titles](./single_think.py)                | 35.5%   | 22.0%  | ***53.0%***  | 38.3%  |
| [T+Genres](./single_think_genres.py)         | ***41.1%***   | 36.4%  | 40.2%  | 34.6%  |
| [T+G+Plot](./single_think_full.py)    | ***43.9%***   | 30.8%  | 33.6%  | ***44.9%***  |
| [Standouts+Titles](./single_popularity.py) | **43.9%**    | 21.5%    | ***45.9%*** | 40.3%    |
| [Standouts+T+Genres](./single_popularity_genres.py)      | ***47.7%***   | 32.7%  | 39.3%  | 37.4%  |
| [Standouts+T+G+Plot](./single_popularity_plot.py.log)      | 39.3%   | 28.9%  | 37.4%  | ***42.0%***  |

### Findings
- MD-genres-BinDiv metric correlation with Gemini is **low** when only genres are given
- CB-plot-ILD metric correlation with Gemini is **low** when plot is given