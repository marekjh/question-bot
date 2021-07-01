from questions import *
import nltk
import pickle
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")
    files = load_files(sys.argv[1])
    
    with open(os.path.join("pickles", "words.pickle"), "rb") as f1:
        file_words = pickle.load(f1)
    with open(os.path.join("pickles", "idfs.pickle"), "rb") as f2:
        file_idfs = pickle.load(f2)
    
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

main()