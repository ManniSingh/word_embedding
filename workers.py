def getTuples(sent,WINDOW=3):
    '''
    Generate tuples in a WINDOW.
    
    parameters:
    -----------
    sent: list of str 
    WINDOW: int
    
    returns:
    --------
    pres: list of sorted tuples
    '''
    pres = list() #partial result
    for i,target in enumerate(sent):
        if len(sent)>=i+WINDOW+1:
            right = sent[i+1:i+WINDOW+1]
            for con in right: 
                if con != target:
                    tup = [con,target]
                    tup.sort()
                    pres.append(tuple(tup))
    return pres