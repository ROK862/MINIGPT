## MiniGPT

MiniGPT is a command-line application that uses the GPT-3.5 language model to answer questions based on a given corpus of information. It leverages natural language processing techniques to identify relevant files and sentences that contain the answers to user queries.

### Technology Stacks

MiniGPT is implemented in Python and utilizes the following technologies:

- **Python**: The application is written in Python, a powerful and versatile programming language.
- **NLTK**: The Natural Language Toolkit (NLTK) library is used for various NLP tasks, including tokenization and sentence tokenization.
- **Math**: The `math` module is used for mathematical calculations, such as computing IDF values.
- **String**: The `string` module provides a set of constants and functions for working with strings, primarily used for removing punctuation.
- **OS**: The `os` module is used for interacting with the operating system, specifically for loading files from a directory.
- **Sys**: The `sys` module provides access to some variables used or maintained by the interpreter and to functions that interact with the Python runtime, in this case, for handling command-line arguments.

### Functionality

1. **Load Files**: The application loads a directory containing `.txt` files and reads their contents into memory, creating a dictionary mapping filenames to their respective contents.

2. **Tokenization**: The `tokenize` function processes a document (string) by converting all words to lowercase and removing punctuation and English stopwords. It uses NLTK's `word_tokenize` function for tokenization.

3. **Compute IDF Values**: The `compute_idfs` function calculates the IDF (Inverse Document Frequency) values for words in a given dictionary of documents. It counts the number of documents that contain each word and applies the IDF formula to compute the IDF value for each word.

4. **Top File Matches**: Given a query, the application determines the top file matches according to TF-IDF (Term Frequency-Inverse Document Frequency) ranking. It calculates the TF-IDF score for each word in the query within each file, and then ranks the files based on the cumulative score.

5. **Extract Sentences**: The application extracts sentences from the top matching files. It tokenizes the passages into sentences using NLTK's sentence tokenizer, removes any empty or non-tokenized sentences, and stores the sentences along with their tokenized form.

6. **Top Sentence Matches**: Given a query, the application identifies the top sentence matches based on IDF ranking. It calculates the IDF score for each query term in each sentence, considers the query term density (the proportion of query terms in the sentence), and ranks the sentences accordingly.

7. **Display Results**: The application displays the top matching sentences to the user.

### Usage

To use MiniGPT, execute the following command:


- `corpus_directory` is the path to the directory containing the corpus of `.txt` files.

The application prompts the user to enter a query and then provides the relevant sentences that answer the query, based on the given corpus and the implemented ranking algorithms.

Note: The application assumes that NLTK's stopwords corpus has been downloaded.

MiniGPT provides a simple and efficient way to process textual data and retrieve answers to user queries from a given corpus.
