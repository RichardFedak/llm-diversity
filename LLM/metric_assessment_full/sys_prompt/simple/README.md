| Evaluation             | Accuracy | cf_ild  | cb_ild  | bin_div  |
|------------------------|----------|--------|--------|---------|
| [Titles](./single_think.py)                | 35.5%   | 22.0%  | ***53.0%***  | 38.3%  |
| [Titles+Genres](./single_think_genres.py)         | ***41.1%***   | 36.4%  | 40.2%  | 34.6%  |
| [Titles+Genres+Plot](./single_think_full.py)    | ***43.9%***   | 30.8%  | 33.6%  | ***44.9%***  |
| [Standouts prompt+Titles](./single_popularity.py)      | ***44.9%***   | 20.6%  | ***43.0%***  | ***43.0%***  |
| [Standouts+Genres](./single_popularity_genres.py)      | ***47.7%***   | 32.7%  | 39.3%  | 37.4%  |
| [Standouts+Genres+Plot](./single_popularity_plot.py.log)      | 39.3%   | 28.9%  | 37.4%  | ***42.0%***  |