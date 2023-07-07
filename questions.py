import sys
import os
import math
import string
import nltk
from nltk.corpus import stopwords


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


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
    files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), encoding='utf8', errors='ignore') as f:
                files[filename] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # Get the stopwords and punctuation.
    sw = stopwords.words("english")
    pc = string.punctuation

    # Use nltk's word_tokenize function to perform tokenization
    tokens = nltk.word_tokenize(document.lower())

    # Remove any tokens that are only punctuation symbols
    tokens = [token for token in tokens if token not in pc]

    # Remove stopwords
    tokens = [
        token for token in tokens if token not in sw]

    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # count the number of documents that contain each word
    word_counts = {}
    for doc in documents.values():
        for word in set(doc):
            # Keep track of frequency of occurrence.
            word_counts[word] = word_counts.get(word, 0) + 1

    # compute the IDF for each word
    num_docs = len(documents)
    idfs = {}
    for word, count in word_counts.items():
        # log(num_docs / numDocsContaining(word))
        idf = math.log(num_docs / count)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    file_scores = {}
    for filename, words in files.items():
        # Calculate TF-IDF score for each word in the query
        score = 0
        for word in query:
            if word in words:
                tf = words.count(word)
                score += tf * idfs[word]
        file_scores[filename] = score

    # Sort files by score and return the top `n` files
    return sorted(file_scores.keys(), key=lambda x: file_scores[x], reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    scores = {}
    for sentence in sentences:
        # Calculate the IDF score of each query term in the sentence
        query_words = query.intersection(sentences[sentence])
        if not query_words:
            continue
        score = sum([idfs[word] for word in query_words])

        # Prefer sentences with a higher query term density in case of ties
        density = len(query_words) / len(sentences[sentence])
        score_density = score * density

        scores[sentence] = (score, density, score_density)

    # Sort the sentences by their scores
    sorted_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Return the top n sentences
    return [sentence for sentence, _ in sorted_sentences[:n]]


if __name__ == "__main__":
    main()
