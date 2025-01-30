import google.generativeai as genai

model = genai.GenerativeModel("models/gemini-1.5-flash")

prompt = """You are an evaluator. You are given three lists of movies. Your task is to order the lists from least to most diverse based on the information given.

Focus on the following two aspects:
1. Genres - This is the most important factor in determining diversity.
2. Plot and Themes - Only if the genres are almost the same or overlap significantly between the lists, use plot and themes to differentiate them.

1. Focus on Genres:
   - Assess the variety and uniqueness of genres in each list.
   - Identify how diverse the genres are within each list and across lists.

2. Focus on Plot and Themes:
   - If genres are similar or overlap significantly between the lists:
    - Analyze the plots to find nuanced differences in storytelling approaches.
    - Evaluate the overarching themes and how they contribute to the distinctiveness of each list.

List A:
- Lord of the Rings: The Return of the King, The (2003) - Genres of the movie: Action, Adventure, Drama, Fantasy - Plot of the movie: Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring.
- Argo (2012) - Genres of the movie: Biography, Drama, Thriller - Plot of the movie: Acting under the cover of a Hollywood producer scouting a location for a science fiction film, a CIA agent launches a dangerous operation to rescue six Americans in Tehran during the U.S. hostage crisis in Iran in 1979.
- Bridget Jones's Diary (2001) - Genres of the movie: Comedy, Drama, Romance - Plot of the movie: Bridget Jones is determined to improve herself while she looks for love in a year in which she keeps a personal diary.
- Finding Nemo (2003) - Genres of the movie: Animation, Adventure, Comedy, Family - Plot of the movie: After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home.
- V for Vendetta (2006) - Genres of the movie: Action, Drama, Sci-Fi, Thriller - Plot of the movie: In a future British dystopian society, a shadowy freedom fighter, known only by the alias of ""V"", plots to overthrow the tyrannical government - with the help of a young woman.
- Forrest Gump (1994) - Genres of the movie: Drama, Romance - Plot of the movie: The history of the United States from the 1950s to the '70s unfolds from the perspective of an Alabama man with an IQ of 75, who yearns to be reunited with his childhood sweetheart.
- Incredibles, The (2004) - Genres of the movie: Animation, Action, Adventure, Family - Plot of the movie: While trying to lead a quiet suburban life, a family of undercover superheroes are forced into action to save the world.

List B:
- Lord of the Rings: The Return of the King, The (2003) - Genres of the movie: Action, Adventure, Drama, Fantasy - Plot of the movie: Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring.
- V for Vendetta (2006) - Genres of the movie: Action, Drama, Sci-Fi, Thriller - Plot of the movie: In a future British dystopian society, a shadowy freedom fighter, known only by the alias of ""V"", plots to overthrow the tyrannical government - with the help of a young woman.
- Argo (2012) - Genres of the movie: Biography, Drama, Thriller - Plot of the movie: Acting under the cover of a Hollywood producer scouting a location for a science fiction film, a CIA agent launches a dangerous operation to rescue six Americans in Tehran during the U.S. hostage crisis in Iran in 1979.
- Donnie Darko (2001) - Genres of the movie: Drama, Mystery, Sci-Fi, Thriller - Plot of the movie: After narrowly escaping a bizarre accident, a troubled teenager is plagued by visions of a man in a large rabbit suit who manipulates him to commit a series of crimes.
- Fugitive, The (1993) - Genres of the movie: Action, Crime, Drama, Mystery, Thriller - Plot of the movie: Dr. Richard Kimble, unjustly accused of murdering his wife, must find the real killer while being the target of a nationwide manhunt led by a seasoned U.S. Marshal.
- Shawshank Redemption, The (1994) - Genres of the movie: Drama - Plot of the movie: Over the course of several years, two convicts form a friendship, seeking consolation and, eventually, redemption through basic compassion.
- Illusionist, The (2006) - Genres of the movie: Drama, Fantasy, Mystery, Romance, Thriller - Plot of the movie: In turn-of-the-century Vienna, a magician uses his abilities to secure the love of a woman far above his social standing.

List C:
- Doug Stanhope: Oslo - Burning the Bridge to Nowhere (2011) - Genres of the movie: Comedy - Plot of the movie: 
- El Niño (2014) - Genres of the movie: Action, Adventure, Crime, Drama, Thriller - Plot of the movie: A small-time trafficker working in the Gibraltar Straits.
- Patton Oswalt: I Love Everything (2020) - Genres of the movie: Comedy - Plot of the movie: Turning 50. Finding love again. Buying a house. Experiencing existential dread at Denny's. Life comes at Patton Oswalt fast in this stand-up special.
- Dabangg 2 (2012) - Genres of the movie: Action, Comedy, Crime - Plot of the movie: Chulbul Pandey invites a fresh trouble when he kills the brother of a notorious politician and the former swears to wreak havoc in his life.
- Pokémon the Movie: White - Victini and Zekrom (2011) - Genres of the movie: Animation, Adventure, Drama, Family, Fantasy, Sci-Fi - Plot of the movie: The greatest adventure in Pokémon history approaches.
- Saturday Night Live: The Best of Will Ferrell (2002) - Genres of the movie: Documentary, Comedy - Plot of the movie: The best skits from Will Ferrell's days on Saturday Night Live 1995-2002
- Joan Didion: The Center Will Not Hold (2017) - Genres of the movie: Documentary, Biography - Plot of the movie: Literary icon Joan Didion reflects on her remarkable career and personal struggles in this intimate documentary directed by her nephew, Griffin Dunne.

For example, if the least diverse list is B, then C, and the most diverse is A, the output should be: B,C,A

Do NOT provide explanations, comments, or any additional text — only the final output in the required format representing the order from least to most diverse list.

"""

print("total_tokens: ", model.count_tokens(prompt))

# LONGEST prompt ~2k tokens