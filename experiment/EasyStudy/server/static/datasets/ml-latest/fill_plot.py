import pandas as pd
import json

#https://osf.io/chbj9/files/osfstorage/675a1d242d78acf176ea0433

with open('movie_data_plot.json', 'r', encoding='utf-8') as f:
    plot_data = json.load(f)

df = pd.read_csv('movies.csv')

def get_plot(movie_id):
    movie_str_id = str(movie_id)
    if movie_str_id in plot_data:
        plot_list = plot_data[movie_str_id].get("plot", [])
        if plot_list:
            return plot_list[0]
    return "X"

df['plot'] = df['movieId'].apply(get_plot)
df.to_csv('movies_plot.csv', index=False)
