import os
import sys
import argparse
import collections
import warnings
import numpy
import pickle

BOF = 0
EOF = 1
#Vocabulary is stored in an array, each word is identified by an index.
vocabulary = ['BOF', 'EOF']
#Probabilities are stored in a list of ngram models.
ngrams = []

#Load hmm data from a binary file.
def load(filename):
    global vocabulary
    global ngrams
    f = open(filename, 'rb')
    vocabulary, ngrams = pickle.load(f)

#Output hmm data to a binary file.
def save(filename):
    f = open(filename, 'wb')
    pickle.dump((vocabulary, ngrams), f)

#Probabilistically generate text from the model.
def generate():
    output = [BOF]
    vocab_size = len(vocabulary)
    while output[-1] != EOF:
        probs = [probability(i, output) for i in range(1, vocab_size)]
        s = sum(probs)
        probs = [i/s for i in probs]
        next_word = numpy.random.choice(vocab_size-1, 1, p=probs)[0]+1
        output.append(next_word)
    string_output = ""
    for word in output[1:-1]:
        string_output += vocabulary[word] + " "
    print(string_output)

#Find the probability of a next word given current state.
def probability(word, state):
    unk = 1e-10#1.0/len(vocabulary) #base probability for unknowns
    try:
        probability = ngrams[0][word] #simple word probability
    except:
        return unk
    for i in range(1, min(len(ngrams)-1, len(state))):
        transitions = ngrams[i+1][word]
        ngram = ngram_helper(state[-1-i:])
        if ngram in transitions:
            probability *= transitions[ngram]
        else:
            probability *= unk
    return probability
        
#Print the model to stdout.
def print_model():
    print("Vocab Size: {0}".format(len(vocabulary)))
    print("n: {0}\n".format(len(ngrams)))
    print("1grams:")
    for i in range(len(vocabulary)):
        try:
            print("\t"+vocabulary[i]+"\t"+str(ngrams[0][i]))
        except:
            print("\t"+vocabulary[i])
    if len(ngrams) > 1:
        for i in range(1, len(ngrams)):
            print(str(i+1)+"grams: ")
            for j in range(len(vocabulary)):
                for k in ngrams[i][j]:
                    print("\t{0} {1}\t{2}".format(k, vocabulary[j], ngrams[i][j][k]))

#This function turns dictionary keys into readable strings 
def ngram_helper(key):
    string = ""
    for token in list(key):
        string += vocabulary[token] + " "
    return string

#Get transition probabilities using given ngram size and data files.
def ngram(n, data):
    if n == 1: #unigrams have different structure than 2+grams
        model = [0.0 for word in vocabulary]
        total = 0
        for tokens in data:
            tokens = tokens
            for token in tokens:
                model[token] += 1.0
                total += 1.0
        for i in range(len(model)):
            model[i] /= total
        return model

    model = [collections.Counter() for word in vocabulary]
    totals = [0.0 for word in vocabulary]
    for tokens in data:
        if len(tokens) >= n:
            for i in range(n-1, len(tokens)):
                model[tokens[i]][ngram_helper(tokens[i+1-n:i])] += 1
                totals[tokens[i]] += 1
        else:
            warnings.warn("File "+filename+" too short to form "
                          +str(n)+"grams.")
    for i in range(len(model)):
        model[i] = dict((key, count/totals[i])
                         for key, count in model[i].items())

    return model

#Initializes global vocabulary variable and returns input data
# in the form of integer sequences.
def vocab(files):
    global vocabulary
    vocab = {'BOF':0, 'EOF':1}
    data = []

    for filename in files:
        data.append([])
        tokens = []
        f = open(filename)
        for line in f.readlines():
            if len(line) > 1:
                tokens += ['BOF'] + line.split() + ['EOF']
        f.close()
        for token in tokens:
            if token in vocab:
                idx = vocab[token]
            else:
                idx = len(vocabulary)
                vocab[token] = idx
                vocabulary.append(token)
            data[-1].append(idx)
    return data

def main():
    #Parse the arguments.
    parser = argparse.ArgumentParser(description = 'An HMM for plain text.')
    parser.add_argument('--printout', action='store_true',
                        help='print the model to stdout')
    parser.add_argument('--generate', metavar="l", type=int,
                        help='generate l lines of text from the model')
    parser.add_argument('--save', help='save hmm to a binary file')
    parser.add_argument('--load', help='load hmm from a binary file')
    parser.add_argument('--input', type=str, nargs='+',
                        help='1+ directories or files with input data')
    parser.add_argument('--ngrams', metavar='n', type=int, default=2,
                        help='maximum ngram length (default 2)')
    args = parser.parse_args()
    file_list = args.input
    n = args.ngrams

    if args.load:
        load(args.load)
    elif args.input:
        #Expand any directories to get complete file list.
        idx = 0
        while (idx < len(file_list)):
            directory = file_list[idx]
            try:
                for filename in os.listdir(directory):
                    file_list.append(os.path.join(directory, filename))
                file_list.pop(idx)
            except:
                idx += 1
        #Get the vocabulary from the files
        data = vocab(file_list)

        #Get ngram probabilities.
        for i in range(1, n+1):
            ngrams.append(ngram(i, data))

    #Optional args
    if args.printout:
        print_model()
    for i in range(args.generate):
        generate()
    if args.save:
        save(args.save)

main()
