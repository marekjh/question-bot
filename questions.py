import nltk
import sys
import os
import math
import string
import collections

FILE_MATCHES = 1
SENTENCE_MATCHES = 3

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    output = {}
    texts = os.listdir(directory)
    for text in texts:
        with open(os.path.join(directory, text), "r", encoding="utf8") as f:
            output[text] = f.read()
    return output




def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    p = string.punctuation
    sw = nltk.corpus.stopwords.words("english")
    tokens = [token.lower() for token in nltk.word_tokenize(document)]
    return [token for token in tokens if token not in p and token not in sw]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    word_lists = documents.values()
    length = len(word_lists)
    words = set()
    word_dicts = []
    # print("Joining words/building word_dict...")
    for word_list in word_lists:
        words = set(word_list).union(words)
        word_dicts.append({word: True for word in word_list})
    # length2 = len(words)
    # print("Done!")
    # print("Iterating through words...")
    # word_count = 0
    for word in words:
        idfs[word] = math.log(length / sum([1 if word_dict.get(word) else 0 for word_dict in word_dicts]))
        # word_count += 1
        # print(f"{word_count / length2 * 100}%")
    # print("Done!")
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = {}
    for (filename, word_list) in files.items():
        counts = collections.Counter(word_list)
        tfidfs[filename] = 0
        for word in query:
            tfidfs[filename] += idfs[word] * counts[word]
    return sorted(tfidfs, key=lambda k: tfidfs[k], reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ranking = {}
    for (sentence, word_list) in sentences.items():
        idf_sum = sum([idfs[word] for word in word_list if word in query])
        term_density = sum([1 if word in query else 0 for word in word_list]) / len(word_list)
        ranking[sentence] = (idf_sum, term_density)
    ordering = sorted(ranking, key=lambda k: ranking[k][1], reverse=True)
    ordering.sort(key=lambda k: ranking[k][0], reverse=True)
    return ordering[:n]




if __name__ == "__main__":
    main()
