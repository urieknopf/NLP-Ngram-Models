# prog2 n-gram language modeling Uriel Knopf
import re
import math
import random
from collections import Counter
output = open("ngram_program_output.txt", "w")


def read_sentences(file):
    with open(file, "r") as f:
        return [re.split("\s+", line.lower().rstrip('\n')) for line in f]


# ################ UNIGRAM CALCULATION FUNCTIONS ##################
def unigram(unigram):
    count = word_count[unigram]
    return count / num_of_words


def unigram_laplace(unigram):
    count = word_count[unigram] + 1
    return count / (num_of_words + volume_words)


def unigram_sentence(sentence):
    sentence_words = sentence.lower().split()
    p = 0
    for words in sentence_words:
        if p != 0:
            p *= unigram(words)
        else:
            p = unigram(words)
    p = math.log(p, 10)
    return p


def unigram_sentence_laplace(sentence):
    sentence_words = sentence.lower().split()
    p = 0
    for words in sentence_words:
        if p != 0:
            p *= unigram_laplace(words)
        else:
            p = unigram_laplace(words)
    p = math.log(p, 10)
    return p


def unigram_laplace_sentence_generation(sentence_length):
    random_sentence = "<s> "
    for words in range(sentence_length):
        p = random.randint(1, len(unigram_list)-2)
        random_sentence += (unigram_list[p][0] + " ")
        if unigram_list[p][0] == "</s>":
            return random_sentence
    random_sentence += "</s>"
    return random_sentence


# ################ BIGRAM CALCULATION FUNCTIONS ###################
def bigram(word1, bigram_count):
    return (1.0 * bigram_count) / word_count[word1]


def bigram_laplace(word1, bigram_count):
    return (1.0 * (bigram_count + 1)) / (word_count[word1] + volume_words)


def bigram_sentence(sentence):
    sentence_words = sentence.lower().split()
    p = 1
    for words in range(len(sentence_words) - 1):
        found = 0
        current_sentence_bigram = (sentence_words[words], sentence_words[words + 1])
        for ro in range(len(bigram_list)):
            if bigram_list[ro][0] == "0":
                break
            else:
                if current_sentence_bigram == (bigram_list[ro][0], bigram_list[ro][1]):
                    p *= bigram_list[ro][3]
                    found = 1
                    break
        if found == 0:
            p *= 0
            return "-Infinity"
    return "{:.4f}".format(math.log(p, 10))


def bigram_sentence_laplace(sentence):
    sentence_words = sentence.lower().split()
    p = 1
    for words in range(len(sentence_words) - 1):
        found = 0
        current_sentence_bigram = (sentence_words[words], sentence_words[words+1])
        for ro in range(len(bigram_list)):
            if bigram_list[ro][0] == "0":
                break
            else:
                if current_sentence_bigram == (bigram_list[ro][0], bigram_list[ro][1]):
                    p *= bigram_list[ro][4]
                    found = 1
                    break
        if found == 0:
            p *= bigram_laplace(sentence_words[words], 0)
    return "{:.4f}".format(math.log(p, 10))


def bigram_laplace_sentence_generation(sentence_length):
    random_sentence, previous_word = "<s> ", "<s>"
    for words in range(sentence_length):
        highest = -1
        if previous_word == "</s>":
            return random_sentence
        elif previous_word == "<s>":
            p = random.randint(1, len(unigram_list) - 2)
            current_word = unigram_list[p][0]
        else:
            for ro in range(len(bigram_list)):
                if bigram_list[ro][0] == "0":
                    break
                else:
                    if previous_word == bigram_list[ro][0]:
                        if highest < bigram_list[ro][4]:
                            highest = bigram_list[ro][4]
                            current_word = bigram_list[ro][1]
        random_sentence += (current_word + " ")
        previous_word = current_word
    random_sentence += "</s>"
    return random_sentence


# ################ Sentence Generation and Output Functions #########
def random_sentences(type):
    output.write("\nRandom sentences from " +type + " language model with laplace smoothing:\n")
    for x in range(1, 11):
        num = random.randint(0, 15)
        if type == "unigram":
            sentence = unigram_laplace_sentence_generation(num)  # generate random sentences
        else:
            sentence = bigram_laplace_sentence_generation(num)  # generate random sentences
        output.write("{0:<4}".format(str(x) + ": "))  # print sentence
        output.write(sentence)
        output.write("\n")


def bigram_output(type, n):
    output.write("\nBigram " + type + "Language Model:\n")
    for ro in range(bigram_rows):
        if bigram_list[ro][0] == "0":
            break
        else:
            bigram_output = "[" + bigram_list[ro][0] + ", " + bigram_list[ro][1] + "]"
            print_ngram(bigram_output, bigram_list[ro][2], bigram_list[ro][3+n])


def unigram_output(unigram_type, n):
    output.write(unigram_type + "Language Model:\n")
    for rowz in range(volume_words):
        if unigram_list[rowz][0] == "0":
            break
        else:
            print_ngram(unigram_list[rowz][0], unigram_list[rowz][1], unigram_list[rowz][2+n])


def print_ngram(current_word, count, prob):
    output.write("{0:<16}".format(current_word))
    output.write("{0:<4}".format(count))
    output.write("{:.4f}".format(prob))
    output.write("\n")


def print_sentence_probabilities(sent, id):  # Generate and print sentence probabilities
    output.write("{0:<15}".format("Sentence" + id))
    output.write("{0:<10}".format("{:.4f}".format(unigram_sentence(sent))))
    output.write("{0:<10}".format("{:.4f}".format(unigram_sentence_laplace(sent))))
    output.write("{0:<10}".format(bigram_sentence(sent)))
    output.write("{0:<10}".format(bigram_sentence_laplace(sent)))
    output.write("\n")


# ###################### MAIN ######################################
if __name__ == '__main__':
    #  Read in and append <s> </s> to sentences  #
    dataset = read_sentences("training_data.txt")
    word_list = []
    for t in dataset:
        if t[0] != "<s>":
            t.insert(0, "<s>")
        length = len(t)
        if t[length - 1] != "</s>":
            t.append("</s>")
        for s in t:
            word_list.append(s)

# #### calc word count, total words, + volume #####
    word_count = Counter(word_list)
    num_of_words = sum(word_count.values())
    volume_words = len(word_count)

# #### calc /print unigram and unig laplace ####
    unigram_cols, uni_row = 4, 0
    unigram_list = [["0" for i in range(unigram_cols)] for j in range(volume_words)]

    for key, value in word_count.items():
        unigram_list[uni_row][0] = key
        unigram_list[uni_row][1] = value
        unigram_list[uni_row][2] = unigram(key)
        unigram_list[uni_row][3] = unigram_laplace(key)
        uni_row += 1

    unigram_output("Unigram ", 0)
    unigram_output("\nUnigram laplace ", 1)

# ########## 2d Bigram list each row: [w1][w2][c][p]#########
    bigram_cols, bigram_rows = 5, (num_of_words - len(dataset))  # rows = there is one less bi/sent than word/sent
    bigram_list = [["0" for i in range(bigram_cols)] for j in range(bigram_rows)]
    current_bigram = ("idk", "error?")  # initialized? eh

    for sentence in range(len(dataset)):
        for word in range(len(dataset[sentence]) - 1):
            found = 0
            current_bigram = (dataset[sentence][word], dataset[sentence][word + 1])
            # print(current_bigram) # just here for testing
            for row in range(bigram_rows):
                if bigram_list[row][0] == "0":
                    break
                else:
                    if current_bigram == (bigram_list[row][0], bigram_list[row][1]):
                        bigram_list[row][2] += 1
                        found = 1
                        break
            if found == 0:
                for y in range(bigram_rows):
                    if bigram_list[row][0] == "0":
                        bigram_list[row][0], bigram_list[row][1] = current_bigram
                        bigram_list[row][2] = 1

    for ro in range(bigram_rows):
        if bigram_list[ro][0] == "0":
            break
        else:
            bigram_list[ro][3] = bigram(bigram_list[ro][0], bigram_list[ro][2])
            bigram_list[ro][4] = bigram_laplace(bigram_list[ro][0], bigram_list[ro][2])

# ########### print bigram and bigram laplace #############
    bigram_output("", 0)
    bigram_output("laplace ", 1)

# ############# Sentence Probabilities ################
    s1, s2 = "<s> take the block on the green circle </s>", "<s> put the block on the circle on the red circle </s>"
    output.write("{0:<15}".format("\nSent. label    Unigram   UnigramL  Bigram    BigramL\n"))  # probability labels
    print_sentence_probabilities(s1, "1")
    print_sentence_probabilities(s2, "2")

# ############### Random Sentences ######################
    random_sentences("unigram")
    random_sentences("bigram")

    print("Program run successful.")
    output.close()
