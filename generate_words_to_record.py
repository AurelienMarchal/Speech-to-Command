from ngrams import count_ngrams_file

command_list_file = "full_command_list.txt"


unigrams = count_ngrams_file(command_list_file, 1)

bigrams = count_ngrams_file(command_list_file, 2)

print(bigrams)