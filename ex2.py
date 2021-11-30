import sys



def init(params):
    with open(output, 'w') as f:
        for i in range(6):
            f.write(f'Output{i + 1}: {params[i]}\n')


def development_pre_processing():
    # take all the articles from the develop file (i%4 == 2 because [0 -> header, 1-> \n ,2->article,3->\n]
    with open(development, 'r') as f:
        lines = [l for i, l in enumerate(f.readlines()) if i % 4 == 2]

    # flat the lines to all the words in the file
    words = [word for line in lines for word in line.split()]

    with open(output, 'a') as f:
        f.write(f'Output{7}: {len(words)}\n')
    return words

def separate_validation(words,div = 0.9):
    train_len = int(len(words) * div)
    return words[:train_len], words[train_len:]


def lidstone_model_training(words):
    train, validate = separate_validation(words)
    set_train = set(train)
    word_count = train.count(word)
    word_mle = word_count/len(train)
    unseen_mle = 0/len(train)
    lamda = 0.1
    word_lidstone = (lamda + word_count)/(lamda*V + len(train))
    unseen_lidstone = lamda/(lamda*V + len(train))
    params = [len(validate),len(train),len(set_train), word_count,word_mle,unseen_mle,word_lidstone,unseen_lidstone]
    with open(output, 'a') as f:
        for i in range(8,16):
            f.write(f'Output{i}: {params[i-8]}\n')


if __name__ == '__main__':
    V = 300000
    development, test, word, output = sys.argv[1:]
    init([development, test, word, output, V, 1 / V])
    words = development_pre_processing()
    lidstone_model_training(words)
