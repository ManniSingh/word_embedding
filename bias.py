from nltk.corpus import wordnet as wn

def get_biases(ss):
    text=ss.definition()
    doc = nlp(text)
    candidates = set()
    for word in doc:
        if word.pos_=='NOUN':
            #print(word.text)
            synsets = wn.synsets(word.text)
            for _b in synsets:
                sim=a.wup_similarity(_b)
                if sim>0.5:
                    candidates.add(word.text)
    if len(candidates)>0:  
        return b
    else:
        return None