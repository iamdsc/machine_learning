# Preprocessing documents
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def tokenizer(text):
    """" Splitting at white spaces """
    return text.split()

porter=PorterStemmer()
def tokenizer_porter(text):
    """ Applying Porter Stemming Algorithm """
    return [porter.stem(word) for word in text.split()]

print(tokenizer('runners like running and thus they run'))
print(tokenizer_porter('runners like running and thus they run'))

# Removing stop words
stop=stopwords.words('english')
print([w for w in tokenizer_porter('runners like running and thus they run')[-10:] if w not in stop])
