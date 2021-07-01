import sys
import os
import pickle
from questions import *


if len(sys.argv) != 2:
    sys.exit("Usage: python dump.py corpus")

files = load_files(sys.argv[1])
print("Tokenizing files...")
file_words = {
    filename: tokenize(files[filename])
    for filename in files
}
print("Done!")
print("Computing idfs...")
file_idfs = compute_idfs(file_words)

with open(os.path.join("pickles", "words.pickle"), "wb") as f1:
    pickle.dump(file_words, f1, pickle.HIGHEST_PROTOCOL)

with open(os.path.join("pickles", "idfs.pickle"), "wb") as f2:
    pickle.dump(file_idfs, f2, pickle.HIGHEST_PROTOCOL)