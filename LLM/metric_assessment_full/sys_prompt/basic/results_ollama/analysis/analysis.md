# Small context problem ?

For short prompts, the Llama *mostly* works OK:

```json
[
    {
        "file": "genres.json",
        "correct": 107,
        "incorrect": 0,
        "incorrect_outputs": []
    },
    {
        "file": "genres_plot.json",
        "correct": 105,
        "incorrect": 2,
        "incorrect_outputs": [
            "List_C",
            "List_A"
        ]
    },
```

Plot in prompt causes problems, as it takes up the context window and model starts to generate output, that is different from the expected one (but still menegable to fix in post-processing):

``` json
        "file": "standouts_title_genres_plot.json",
        "correct": 57,
        "incorrect": 50,
        "incorrect_outputs": [
            "List A",
            "List C",
            "List A",
            "List C",
            ...
```

However there are some (few) cases, where the model just fails to produce output that is even close to the expected one:


```json


        "file": "summary_genres_plot.json",
        "correct": 24,
        "incorrect": 83,
        "incorrect_outputs": [
            "List B",
            "List C",
            "List C",
            "Most diverse list selected from the given options",
            ...
            "Although List A has some repeated genres like Action and Adventure, it has a diverse set of movies that cover a wide range of themes. For instance, the Harry Potter series explores magic and friendship, while The Martian is a sci-fi movie about survival. List B is also quite diverse but it leans more towards superhero movies. List C is the most predictable with a heavy focus on fantasy movies like Harry Potter and Narnia.",
            ...
            Or even...
            ...
            "The most diverse list is difficult to determine due to similar levels of diversity across all three lists, A, B, and C.",

```

