# hmm
Hidden Markov Model implementation in Python

## Version
This code is designed to run with Python 2.7

## How to use
The input text files should already be tokenized with whitespace between tokens.

Usage example:
```
python hmm.py --input input.txt --ngrams 3 --save example.dat
python hmm.py --load example.dat --generate 10 > output.txt
```
The above would create a trigram hidden markov model with input.txt, then use the model to generate 10 lines of text.

Alternatively:
```
python hmm.py --input input.txt --ngrams 3 --generate 10 > output.txt
```
