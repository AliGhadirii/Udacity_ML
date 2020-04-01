
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
print(stopwords.words('english'))
stemmer = SnowballStemmer('english')
print(stemmer.stem('responsivity'))
