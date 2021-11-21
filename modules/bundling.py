import pandas as pd


def bundle_data(pos,neg):
    g_emot = ['data/goemotions_2.csv','data/goemotions_3.csv']
    df = pd.read_csv('data/goemotions_1.csv')
    for g in g_emot:
        tdf = pd.read_csv(g)
        df = df.append(tdf, ignore_index=True)
    del df['id']
    del df['author']
    del df['subreddit']
    del df['link_id']
    del df['parent_id']
    del df['created_utc']
    del df['rater_id']
    del df['example_very_unclear']
    dic= {}
    k=1
    for d in df.columns[1:]:
        if d in pos:
            dic[k] = 'p'
        else:
            dic[k] = 'n'
        k+=1
    df['positive'] = [0]*len(df)
    df['negative'] = [0]*len(df)
    dummy = df.values
    for row in dummy:
        try:
            idx = list(row).index(1)
        except:
            idx = -1
        if idx !=-1:
            if dic[idx] == 'p':
                row[-1] = 1
            else:
                row[-2] = 1
    df = pd.DataFrame(dummy,columns=df.columns)
    df = df.filter(['text','positive','negative'])
    df = df[df['positive'] != df['negative']]
    return df