# No title

| Name               | llm-user accuracy | llm-CF_ILD Accuracy | llm-CB_ILD Accuracy | llm-BIN_DIV Accuracy | EMD    | Spearman |
|--------------------|-----------------|-----------------|-----------------|-------------------|--------|----------|
| genres            | **26.13%**          | 27.48%          | 26.58%          | 29.28%          | 0.23   | 0.14   |
| plot   | 18.92%          | 20.27%          | 21.62%          | 18.92%          | 0.23   | 0.06   |
| genres_plot       | 20.27%          | 21.62%          | 23.87%          | 25.23%          | 0.24   | 0.09   |

# With title

| Name           | llm-user accuracy | llm-CF_ILD Accuracy | llm-CB_ILD Accuracy | llm-BIN_DIV Accuracy | EMD    | Spearman |
|----------------|-----------------|-----------------|-----------------|-------------------|--------|----------|
| title_genres | **24.77%**          | 26.13%          | 23.87%          | **31.53%**          | 0.25   | 0.16   |
| title_genres_plot| 22.97%          | 25.23%          | 26.58%          | 26.58%          | 0.24   | 0.09   |
| title            | 21.62%          | 24.32%          | 22.97%          | 21.62%          | 0.22   | 0.08   |
| title_plot       | 22.07%          | 23.42%          | 27.48%          | 22.52%          | 0.24   | 0.09   |

# Summary

| Name                   | llm-user accuracy | llm-CF_ILD Accuracy | llm-CB_ILD Accuracy | llm-BIN_DIV Accuracy | EMD    | Spearman |
|------------------------|-----------------|-----------------|-----------------|-------------------|--------|----------|
| summary_genres       | **24.32%**          | 24.77%          | 27.03%          | 27.48%          | 0.27   | 0.13   |
| summary_genres_plot  | 22.52%          | 18.02%          | 21.17%          | 26.13%          | 0.27   | 0.08   |
| summary_plot           | 21.62%          | 19.37%          | 25.23%          | 23.42%          | 0.27   | 0.10   |
| summary_title_genres   | 22.52%          | 25.68%          | 24.32%          | 26.13%          | 0.27   | 0.09   |
| summary_title_genres_plot | 21.62%          | 20.27%          | 23.42%          | 22.07%          | 0.27   | 0.09   |
| summary_title          | 22.97%          | 22.52%          | 22.97%          | 27.48%          | 0.27   | 0.09   |
| summary_title_plot     | 20.72%          | 23.42%          | 25.68%          | 22.52%          | 0.27   | 0.07   |

# Standouts

| Name                        | llm-user accuracy | llm-CF_ILD Accuracy | llm-CB_ILD Accuracy | llm-BIN_DIV Accuracy | EMD    | Spearman |
|-----------------------------|-----------------|-----------------|-----------------|-------------------|--------|----------|
| standouts_genres_plot       | 22.52%          | 26.58%          | 27.48%          | 25.23%          | 0.24   | 0.09   |
| standouts_genres            | **29.28%**          | 29.28%          | 29.28%          | **30.63%**          | 0.22   | 0.17   |
| standouts_plot              | 20.72%          | 27.03%          | 27.48%          | 22.97%          | 0.23   | 0.04   |
| standouts_title             | **24.77%**          | 27.48%          | **30.63%**          | 22.97%          | 0.22   | 0.08   |
| standouts_title_genres      | **24.32%**          | **34.68%**          | **33.78%**          | 29.73%          | 0.24   | 0.17   |
| standouts_title_genres_plot | 21.62%          | 27.03%          | 28.83%          | 22.97%          | 0.23   | 0.10   |
| standouts_title_plot        | 21.62%          | **31.08%**          | **30.63%**          | 23.87%          | 0.23   | 0.13   |