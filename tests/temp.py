import spacy
nlp = spacy.load("en_core_web_sm")

def extract_keywords(query):
    doc = nlp(query)
    keywords = []

    for token in doc:
        # Extract nouns & proper nouns
        if token.pos_ in ["NOUN", "PROPN"]:
            keywords.append(token.text.lower())

    return keywords

print(extract_keywords("I am going for a trip"))
print(extract_keywords("I want clothes for Himachal"))
