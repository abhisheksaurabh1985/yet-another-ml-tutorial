from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
from sklearn.model_selection import train_test_split
import pycrfsuite
import numpy as np
from sklearn.metrics import classification_report


import helper_functions

# Read data file and parse the XML
with codecs.open("../datasets/CRF/reuters.xml", "r", "utf-8") as infile:
    soup = bs(infile, "html5lib")

""" docs[0]: 
[(u'Paxar', 'N'), (u'Corp', 'N'), (u'said', 'I'), (u'it', 'I'), (u'has', 'I'), (u'acquired', 'I'), (u'Thermo-Print', 'N'), (u'GmbH', 'N'), (u'of', 'I'), (u'Lohn', 'N'), (u',', 'I'), (u'West', 'N'), (u'Germany', 'N'), (u',', 'I'), (u'a', 'I'), (u'distributor', 'I'), (u'of', 'I'), (u'Paxar', 'N'), (u'products,', 'I'), (u'for', 'I'), (u'undisclosed', 'I'), (u'terms.', 'I')]
"""
docs = []
for elem in soup.find_all("document"):
    texts = []
    # Loop through each child of the element under "textwithnamedentities"
    for c in elem.find("textwithnamedentities").children:
        if type(c) == Tag:
            if c.name == "namedentityintext":
                label = "N"  # part of a named entity
            else:
                label = "I"  # irrelevant word
            for w in c.text.split(" "):  # Loop through each token within the XML tags and append label as "N"/"I".
                if len(w) > 0:
                    texts.append((w, label))
    docs.append(texts)    


data = []
for i, doc in enumerate(docs):

    # Obtain the list of tokens in the document
    tokens = [t for t, label in doc]

    # Perform POS tagging
    tagged = nltk.pos_tag(tokens)

    # Take the word, POS tag, and its label
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])


X = [helper_functions.extract_features(doc) for doc in data]
y = [helper_functions.get_labels(doc) for doc in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,  

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')


# In[34]:


tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Let's take a look at a random sample in the testing set
i = 12
for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
    print("%s (%s)" % (y, x))


# In[35]:



# Create a mapping of labels to indices
labels = {"N": 1, "I": 0}

# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

# Print out the classification report
print(classification_report(
    truths, predictions,
    target_names=["I", "N"]))

