# %%
import pandas as pd 
import scraper.reddit as sr
import scraper.tweets as st

# %%
subreddits = ['SuicideWatch', 'depression', 'Anxiety']
df1 = pd.DataFrame(data = sr.get_subreddits(subreddits,5000))

# %%
df1.to_csv('./RedditSuicideData.csv',index=False)

# %%
queries = ['suicide','die','end my life']
df2 = pd.DataFrame(data=st.get_tweets(queries,limit=1000))

# %%
df2.to_csv('./TwitterSuicideData.csv',index = False)

# %%
subreddits = ['movies', 'popular', 'books','Jokes']
df3 = pd.DataFrame(data = sr.get_subreddits(subreddits,5000))

# %%
df3.to_csv('./NoSuicideData.csv',index = False)


