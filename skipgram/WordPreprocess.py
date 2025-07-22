import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt

corpus = Korpora.load('nsmc')
corpus = pd.DataFrame(corpus.test)

tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]
print(tokens[:3]) 