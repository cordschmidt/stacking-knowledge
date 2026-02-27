# Number of tokens for each corpus on the train set fpr the 3k tokenizer
DATASET_TOKEN_COUNTS = {
    "childes.train": 6_518_528,
    "bnc_spoken.train": 1_482_112,
    "switchboard.train": 272_000,
    "open_subtitles.train": 3_816_192,
    "simple_wiki.train": 2_936_320,
    "gutenberg.train": 4_277_760,
}


# Difficulty order for spoken-first curriculum
# Lower numbers mean "easier" datasets to start training on
SPOKEN_FIRST_DATASET_ORDER_BABYLM_2023 = {
    "childes.train": 1,
    "bnc_spoken.train": 2,
    "switchboard.train": 2,
    "open_subtitles.train": 3,
    "simple_wiki.train": 4,
    "gutenberg.train": 5,
}

# Difficulty order for grammatical-first curriculum
GRAMMATICAL_FIRST_DATASET_ORDER_BABYLM_2023 = {
    "gutenberg.train": 1,
    "simple_wiki.train": 2,
    "open_subtitles.train": 3,
    "bnc_spoken.train": 4,
    "switchboard.train": 4,
    "childes.train": 5,
}

NUM_STAGES = max(SPOKEN_FIRST_DATASET_ORDER_BABYLM_2023.values())