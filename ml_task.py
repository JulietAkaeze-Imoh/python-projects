# Install the following libraries in the terminal:
# python -m pip install spacy
# python -m spacy download en_core_web_lg
# python -m pip install transformers torch

# Import all the relevant libraries
import spacy
from transformers import pipeline

# Load the models
nlp_spacy = spacy.load("en_core_web_lg")

ner_hf = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Including a language list with popular languages
COMMON_LANGUAGES = [
    "English", "Chinese", "French", "Spanish", "German", "Italian", "Portuguese",
    "Russian", "Japanese", "Korean", "Arabic", "Hindi", "Bengali", "Swahili",
    "Yoruba", "Igbo", "Zulu", "Turkish", "Dutch", "Greek", "Hebrew", "Urdu",
    "Tamil", "Persian", "Thai", "Vietnamese"
]

# Inputting a user query
query = "I want to watch Black Panther in Toronto with English subtitles and Chinese audio"


# spaCy
doc = nlp_spacy(query)

# Hugging Face
hf_entities = ner_hf(query)

# Storing preferences in a user profile dictionary
user_profile = {
    "movies": set(),
    "location": set(),
    "audio": set(),
    "language": set()
}

# Extract from spaCy
for ent in doc.ents:
    if ent.label_ == "WORK_OF_ART":       # Movie titles are often WORK_OF_ART
        user_profile["movies"].add(ent.text)
    elif ent.label_ == "GPE":             # Locations
        user_profile["location"].add(ent.text)
    elif ent.label_ in ["LANGUAGE", "NORP"]: # Languages and Nationalities
        if ent.text.capitalize() in COMMON_LANGUAGES:
            user_profile["language"].add(ent.text.capitalize())


# Extract from Hugging Face
for ent in hf_entities:
    if ent["entity_group"] in ["ORG", "MISC"] and len(ent["word"].split()) > 1:
        # Treat multi-word entities as possible movies
        user_profile["movies"].add(ent["word"])
    elif ent["entity_group"] == "LOC":
        user_profile["location"].add(ent["word"])
    elif ent["word"].capitalize() in COMMON_LANGUAGES:
        user_profile["language"].add(ent["word"].capitalize())


# Rule-based extraction for audio preferences
if "subtitles" in query.lower():
    user_profile["audio"].add("subtitles")
if "audio" in query.lower():
    # Extract the word before "audio" (like "Chinese audio")
    words = query.lower().split()
    idx = words.index("audio")
    if idx > 0:
        lang_candidate = words[idx-1].capitalize()
        if lang_candidate in COMMON_LANGUAGES:
            user_profile["audio"].add(lang_candidate)

# Convert sets to lists for final output
user_profile = {k: list(v) for k, v in user_profile.items()}

print("\n User Profile:", user_profile)