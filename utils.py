import spacy
from spacy.tokens.doc import Doc
from spacy.lang.en import English

__all__ = [
    "ner_tagger",
    "get_nouns",
    "get_verbs",
    "tokenize_english"
]

spacy_english_parser = English()

def get_spacy_model(model: any=None, mode: str="efficiency"):

    if model is None:
        if mode == "efficiency":
            model = spacy.load('en_core_web_sm')
        elif mode == "accuracy":
            model = spacy.load('en_core_web_trf')
        else:
            raise Exception("Invalid spacy mode")

    return model

def ner_tagger(doc: Doc):

    if doc.__class__ is not Doc:
        raise TypeError(f"Expecting spacy.tokens.doc.Doc but receive {doc.__class__}")

    tags = []
    for chunk in doc:
        [tags.append((chunk.text, ner.label_)) for ner in doc.ents if (ner.text == chunk.text)]
    
    return tags

def get_nouns(doc: Doc):

    if doc.__class__ is not Doc:
        raise TypeError(f"Expecting spacy.tokens.doc.Doc but receive {doc.__class__}")

    return [chunk.text for chunk in doc.noun_chunks]

def get_verbs(doc: Doc):

    if doc.__class__ is not Doc:
        raise TypeError(f"Expecting spacy.tokens.doc.Doc but receive {doc.__class__}")

    return [token.lemma_ for token in doc if token.pos_ == "VERB"]

def tokenize_english(text):
    lda_tokens = []
    tokens = spacy_english_parser(text=text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append("URL")
        elif token.orth_.startswith("@"):
            lda_tokens.append("SCREEN_NAME")
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


if __name__ == "__main__":
    sentence = "The DAP political education director said Khairy appears to be “much more” competent than Adham"
    model = get_spacy_model()
    docs = model(sentence)

    print("NER: ", ner_tagger(docs))
    print("Nouns: ", get_nouns(docs))
    print("Verbs: ", get_verbs(docs))
    print("Tokenize english: ", tokenize_english(sentence))