import pandas as pd
import json

#https://osf.io/chbj9/files/osfstorage/675a1d242d78acf176ea0433

with open('movie_data_plot.json', 'r', encoding='utf-8') as f:
    plot_data = json.load(f)

df = pd.read_csv('movies.csv')

def get_plot(row):
    current_plot = str(row.get('plot', '')).strip()
    if current_plot and current_plot != "X":
        return current_plot  # Keep existing plot

    movie_str_id = str(row['movieId'])
    plot_list = plot_data.get(movie_str_id, {}).get("plot", [])
    return plot_list[0] if plot_list else "X"

df['plot'] = df.apply(get_plot, axis=1)

df.to_csv('movies.csv', index=False)