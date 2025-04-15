from imdb import Cinemagoer

x = Cinemagoer()

iron_man_id = 371746

movie = x.get_movie(iron_man_id)

print(movie.data.get("plot",[])[0])