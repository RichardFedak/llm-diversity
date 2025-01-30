# Evaluation Summary

| Evaluation Name                                      | Correct Outputs | Incorrect Outputs | Accuracy  |
|------------------------------------------------------|-----------------|-------------------|-----------|
| [think_single_two_options_titles](./single_think_two_option_titles.py)                      | 40              | 148               | 21.28%    |
| [think_single_two_options_titles_no_instructions](./single_think_two_option_titles_no_instructions.py)      | 47              | 140               | ***25.13%***    |
| [think_single_two_options_no_instructions](./single_think_two_option_no_instructions.py)             | 63              | 125               | ***33.51%***    |
| [more_div_full](./analysis_full_more_diverse_only.py)                                        | 45              | 143               | ***23.94%***    |
| [more_div_titles](./analysis_title_more_diverse_only.py)                                      | 45              | 143               | ***23.94%***    |
| [more_div_think](./analysis_more_diverse_only_think.py)                                       | 44              | 144               | 23.40%    |
| [more_div_think_single](./single_more_diverse_only_think.py)                                | 37              | 151               | 19.68%    |
| [more_div_plot](./analysis_plot_more_diverse_only.py)                                        | 43              | 145               | 22.87%    |
| [more_div_genres](./analysis_genres_more_diverse_only.py)                                      | 44              | 144               | 23.40%    |
| [genres_filtered](./analysis_genres.py)                                      | 51              | 171               | 22.97%    |

# Metric correlations
| Eval Name                              | cf_ild   | cb_ild   | ease_ild | genres   | tags     | bin_div  |
|---------------------------------------|----------|----------|----------|----------|----------|----------|
| [think_single_two_options_titles_no_instructions](./single_think_two_option_titles_no_instructions.py) | ***0.3262***   | 0.2796   | 0.2567   | 0.2742   | 0.1353   | 0.2742   |
| [think_single_two_options_titles_no_instructions](./single_think_two_option_titles_no_instructions.py) | 0.2834   | ***0.3065***   | 0.2727   | 0.2419   | 0.1471   | 0.2581   |
| [more_div_think](./analysis_more_diverse_only_think.py)                         | ***0.3316***   | ***0.3011***   | 0.2674   | 0.2796   | 0.1529   | 0.2957   |
| [more_div_titles](./analysis_title_more_diverse_only.py)                        | ***0.3529***   | ***0.3602***   | 0.2941   | 0.2957   | 0.1529   | 0.2742   |
