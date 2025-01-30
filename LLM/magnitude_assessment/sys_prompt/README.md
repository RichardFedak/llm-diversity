# Evaluation Summary

| Evaluation Name                                      | Correct Outputs | Incorrect Outputs | Accuracy  |
|------------------------------------------------------|-----------------|-------------------|-----------|
| think_single_two_options_titles                      | 40              | 148               | 21.28%    |
| think_single_two_options_titles_no_instructions      | 47              | 140               | ***25.13%***    |
| think_single_two_options_no_instructions             | 63              | 125               | ***33.51%***    |
| more_div                                             | 45              | 143               | 23.94%    |
| more_div_titles                                      | 45              | 143               | 23.94%    |
| more_div_think                                       | 46              | 142               | ***24.47%***    |
| more_div_think_single                                | 37              | 151               | 19.68%    |
| more_div_plot                                        | 43              | 145               | 22.87%    |
| more_div_genres                                      | 44              | 144               | 23.40%    |
| genres_filtered                                      | 51              | 171               | 22.97%    |

# Metric correlations
| Eval Name                              | cf_ild   | cb_ild   | ease_ild | genres   | tags     | bin_div  |
|---------------------------------------|----------|----------|----------|----------|----------|----------|
| think_single_two_options_no_instructions | ***0.3262***   | 0.2796   | 0.2567   | 0.2742   | 0.1353   | 0.2742   |
| think_single_two_options_titles_no_instructions | 0.2834   | ***0.3065***   | 0.2727   | 0.2419   | 0.1471   | 0.2581   |
| more_div_think                         | ***0.3316***   | ***0.3011***   | 0.2674   | 0.2796   | 0.1529   | 0.2957   |
| more_div_titles                        | ***0.3529***   | ***0.3602***   | 0.2941   | 0.2957   | 0.1529   | 0.2742   |
