import re
def get_results(sent):
    sent = ' '.join(sent)
    ents = set(re.findall(r'\[\s[\w\s]+\]',sent))
    _ents = [ent.replace(' ','#E ') for ent in ents]
    ents_ = [ent.replace('[#E','[') for ent in _ents]
    res = list()
    for i,ent in enumerate(ents):
        res.append((ent, ents_[i]))
    return res