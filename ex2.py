import math
import sys
from typing import List

from consts import *


class Util:
    @staticmethod
    def separate_validation(words, div=0.9):
        train_len = int(len(words) * div)
        return words[:train_len], words[train_len:]

    @staticmethod
    def count(text):
        """
        :param text: list of words
        :return: dictionary of words and number of instances per word
        """

        dct = {}
        for word in text:
            if word in dct.keys():
                dct[word] += 1
            else:
                dct[word] = 1
        return dct

    @staticmethod
    def get_events_from_file(file):
        """

        :param file: development/test file name
        :return: list of all the words (events) in the file
        """
        # take all the articles from the file (i%4 == 2 because [0 -> header, 1-> \n ,2->article,3->\n]
        with open(file, 'r') as f:
            lines = [l for i, l in enumerate(f.readlines()) if i % 4 == 2]

        # flat the lines to all the words in the file
        words = [word for line in lines for word in line.split()]

        return words


class OutPut(object):
    def __init__(self, output_file):
        self._output_file = output_file
        self._line_count = 0
        self._write_names()

    def _write_names(self):
        with open(self._output_file, 'w') as f:
            f.write(f"#Students\t{NAMES[0]} {NAMES[1]} {IDS[0]} {IDS[1]}\n")

    def write_output(self, outcome):
        self._line_count += 1
        with open(self._output_file, 'a') as f:
            f.write(f"#Output{self._line_count}\t{outcome}\n")

    def write_table(self, f_lambda, f_H, NrT, tr):
        pass


class Lidstone(object):
    def __init__(self, train: List[str], validate: List[str], test: List[str] = None):
        self._train = train
        self._train_length = len(train)
        self._validate = validate
        self._validate_length = len(validate)
        self.test = test

    @property
    def test(self):
        return self._test

    @property
    def test_length(self):
        return len(self.test)

    def lidstone_per_word(self, word_count, _lambda):
        return (_lambda + word_count) / (_lambda * V + self._train_length)

    def perplexity(self, _lambda, words_instances, test=False):
        log_perplexity = 0
        test_set, test_set_length = (self.test, self.test_length) if test else (self._validate, self._validate_length)

        for w in test_set:
            # the number of instances in the file
            num_of_instances = words_instances[w] if w in words_instances.keys() else 0

            # calc the log of the perplexity for one word with the given lambda
            # if pm != 0:
            log_perplexity += math.log(self.lidstone_per_word(num_of_instances, _lambda), 2)

        return 2 ** (-log_perplexity / test_set_length)

    def lambda_tuning(self, words_instances):
        _lambda = 0
        best_lambda = 0
        best_perplexity = float('inf')
        while _lambda <= 2:
            # self.debug(_lambda)
            _lambda += 0.01
            perplexity = self.perplexity(_lambda, words_instances)
            if perplexity < best_perplexity:
                best_lambda = _lambda
                best_perplexity = perplexity

        return best_lambda, best_perplexity

    def f_lambda(self, r, _lambda):
        return self.lidstone_per_word(r, _lambda) * self._train_length

    def debug(self, _lambda):
        epsilon = 0.00000001
        set_train = Util.count(self._train)
        unseen_words = (V - len(set_train)) * self.lidstone_per_word(0, _lambda)
        for word in set_train:
            unseen_words += self.lidstone_per_word(set_train[word], _lambda)
        if unseen_words > 1 + epsilon or unseen_words < 1 - epsilon:
            raise RuntimeWarning(f"Lidstone probabilistic do not sum to 1 for lambda: {_lambda}")

    @test.setter
    def test(self, value):
        self._test = value


class HeldOut:
    def __init__(self, train_set, held_out_set, v_size=V):
        self.train_set = train_set
        self.held_out_set = held_out_set
        self.v_size = v_size
        self.train_count = Util.count(self.train_set)
        self.held_out_count = Util.count(self.held_out_set)
        self.n_0 = self.v_size - len(self.train_count.keys())
        # calculate t_r, n_r
        self.t_r_dict = {}
        self.n_r_dict = {0: self.n_0}
        for word, r in self.train_count.items():
            if r in self.t_r_dict:
                self.t_r_dict[r] += self.held_out_count[word] if word in self.held_out_count else 0
                self.n_r_dict[r] += 1

            else:
                self.t_r_dict[r] = self.held_out_count[word] if word in self.held_out_count else 0
                self.n_r_dict[r] = 1
        # count the frequency of the words in the held out set but not in the train set.
        self.t_r_dict[0] = sum([f for word, f in self.held_out_count.items() if word not in self.train_count.keys()])
        self.unseen_words = [word for word in self.held_out_set if word not in self.train_count]

    def estimate(self, input_word=None):
        # seen words
        if input_word in self.train_count:
            r = self.train_count[input_word]
            t_r = self.t_r_dict[r]
            n_r = self.n_r_dict[r]
            return (t_r / n_r) / len(self.held_out_set)
        # unseen words (including None object)
        else:
            return len(self.unseen_words) / (self.n_0 * len(self.held_out_set))

    def f_H(self, r):
        return ((self.t_r_dict[r] / self.n_r_dict[r]) / len(self.held_out_set)) * len(self.train_set)

    def perplexity(self, test_set):
        log_perplexity = 0
        for w in test_set:
            log_perplexity += math.log(self.estimate(w), 2)

        return 2 ** (-log_perplexity / len(test_set))

    def debug(self):
        epsilon = 0.00000001
        prob_sum = 0
        # unseen events
        prob_sum += self.n_0 * self.estimate()
        # seen events
        for word in self.train_count:
            prob_sum += self.estimate(word)

        if prob_sum > 1 + epsilon or prob_sum < 1 - epsilon:
            raise RuntimeWarning(f"HeldOut probability do not sum to 1")


def init(params, output_manager):
    # outputs 1 to 6
    for p in params:
        output_manager.write_output(p)


def main():
    development_file, test_file, input_word, output = sys.argv[1:]
    output_manager = OutPut(output)
    # get all the words in the development file
    development_words = Util.get_events_from_file(development_file)

    # separate train and validate
    train, validate = Util.separate_validation(development_words)
    lidstone = Lidstone(train, validate)

    # Output 1-6
    init([development_file, test_file, input_word, output, V, 1 / V], output_manager)

    # Output 7
    output_manager.write_output(len(development_words))

    # Output 8
    output_manager.write_output(len(validate))

    # Output 9
    output_manager.write_output(len(train))

    # Output 10
    train_instances = Util.count(train)
    output_manager.write_output(len(train_instances))  # number of different events in the training set

    # Output 11
    word_count = train.count(input_word)
    output_manager.write_output(word_count)

    # Output 12
    input_word_mle = word_count / len(train)
    output_manager.write_output(input_word_mle)

    # Output 13
    unseen_word_mle = train.count('unseen_word') / len(train)
    output_manager.write_output(unseen_word_mle)

    # Output 14
    output_manager.write_output(lidstone.lidstone_per_word(word_count=word_count, _lambda=0.1))

    # Output 15
    output_manager.write_output(lidstone.lidstone_per_word(word_count=train.count('unseen-word'), _lambda=0.1))

    # Output 16
    words_instances = Util.count(train)
    output_manager.write_output(lidstone.perplexity(_lambda=0.01, words_instances=words_instances))

    # Output 17
    output_manager.write_output(lidstone.perplexity(_lambda=0.1, words_instances=words_instances))

    # Output 18
    output_manager.write_output(lidstone.perplexity(_lambda=1.0, words_instances=words_instances))

    best_lambda, best_perplexity = lidstone.lambda_tuning(words_instances=words_instances)
    # Output 19
    output_manager.write_output(best_lambda)

    # Output 20
    output_manager.write_output(best_perplexity)

    """4. Held out model training"""
    training_set, held_out_set = Util.separate_validation(development_words, 0.5)
    held_out_model = HeldOut(training_set, held_out_set, v_size=V)
    held_out_model.debug()

    # Output 21
    output_manager.write_output(len(training_set))

    # Output 22
    output_manager.write_output(len(held_out_set))

    # Output 23
    output_manager.write_output(held_out_model.estimate(input_word))

    # Output 24
    output_manager.write_output(held_out_model.estimate("unseen-word"))

    # Output 25
    test_words = Util.get_events_from_file(test_file)
    output_manager.write_output(len(test_words))

    # Output 26
    lidstone.test = test_words
    lidstone_perplexity = lidstone.perplexity(best_lambda, words_instances, test=True)
    output_manager.write_output(lidstone_perplexity)

    # Output 27
    held_out_perplexity = held_out_model.perplexity(test_words)
    output_manager.write_output(held_out_perplexity)

    # Output 28
    better_model = 'L' if lidstone_perplexity < held_out_perplexity else 'H'
    output_manager.write_output(better_model)

    # OutPut 29
    table_output = ""
    for i in range(10):
        f_lambda = round(lidstone.f_lambda(i, best_lambda), 5)
        f_H = round(held_out_model.f_H(i), 5)
        NTr = held_out_model.n_r_dict[i]
        tr = held_out_model.t_r_dict[i]
        table_output += f"\n{i}\t {f_lambda} {f_H} {NTr} {tr}"
    output_manager.write_output(table_output)


if __name__ == '__main__':
    main()
