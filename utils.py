import spacy

def ner_tagger(model: any=None, sentence:str="", mode: str="efficiency"):

    if model is None:
        if mode == "efficiency":
            model = spacy.load('en_core_web_sm')
        elif mode == "accuracy":
            model = spacy.load('en_core_web_trf')
        else:
            raise Exception("Invalid mode for name tagger")

    tags = []
    words = model(sentence)
    for word in words:
        [tags.append((word.text, ner.label_)) for ner in words.ents if (ner.text == word.text)]

    return tags


if __name__ == "__main__":
    sentence = "The DAP political education director said Khairy appears to be “much more” competent than Adham."
    print(ner_tagger(sentence=sentence))