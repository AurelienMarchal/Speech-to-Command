from itertools import tee, islice
from collections import Counter
import re

def ngrams(lst, n):
    tlst = lst
    while True:
        a, b = tee(tlst)
        l = tuple(islice(a, n))
        if len(l) == n:
            yield l
            next(b)
            tlst = b
        else:
            break

def count_ngrams_file(file_path, n) -> Counter:

    counter = Counter()

    with open(file_path, 'r') as f:
        for line in f.readlines():
            words = re.findall("\w+", line)
            counter.update(ngrams(words, n))

    return counter


if __name__=='__main__':
    print(count_ngrams_file("./full_command_list.txt", 1))
