# Move the dataset mapping here so it can be accessed without loading the whole package
CUSTOM_STAGED_ORDER = {
    "childes.train": 1,
    "bnc_spoken.train": 2,
    "switchboard.train": 3,
    "open_subtitles.train": 4,
    "simple_wiki.train": 5,
    "gutenberg.train": 6,
}

# Difficulty order for spoken-first curriculum
# Lower numbers mean "easier" datasets to start training on
SPOKEN_FIRST_DATASET_ORDER_BABYLM_2023 = {
    "childes.train": 1,
    "bnc_spoken.train": 2,
    "switchboard.train": 2,
    "open_subtitles.train": 3,
    "qed.train": 3,
    "cbt.train": 4,
    "children_stories.train": 4,
    "simple_wiki.train": 5,
    "wikipedia.train": 6,
    "gutenberg.train": 6,
}

# Difficulty order for grammatical-first curriculum
GRAMMATICAL_FIRST_DATASET_ORDER_BABYLM_2023 = {
    "cbt.train": 1,
    "children_stories.train": 1,
    "simple_wiki.train": 2,
    "wikipedia.train": 3,
    "gutenberg.train": 3,
    "open_subtitles.train": 4,
    "bnc_spoken.train": 5,
    "switchboard.train": 5,
    "qed.train": 6,
    "childes.train": 6,
}

NUM_STAGES = max(CUSTOM_STAGED_ORDER.values())