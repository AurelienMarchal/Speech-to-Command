import itertools
from os import path

"""from ngrams import count_ngrams_file
command_list_file = "full_command_list.txt"
unigrams = count_ngrams_file(command_list_file, 1)
bigrams = count_ngrams_file(command_list_file, 2)
"""

class CommandPart:
    def __init__(self, command, ind, transcription) -> None:
        self.command = command
        self.ind = ind
        self.transcription = transcription
    
    def from_same_recording(self, command_part):
        return self.command == command_part.command
    
    def __str__(self) -> str:
        return self.transcription

    def __repr__(self) -> str:
        return str(self)
    


class Command:
    def __init__(self, command_str=None, parts=None) -> None:
        self.command_str = command_str

        assert command_str is not None or parts is not None, "Command or parts must be an object"

        if parts is None:
            self.assembled = False
            self.parts = []
            for (i, word) in enumerate(command_str.split(' ')):
                self.parts.append(CommandPart(self, i, word))

        else:
            self.parts = parts
            self.assembled = True

        if command_str is None:
            self.command_str = ""
            for part in parts:
                self.command_str += str(part) + " "
            self.command_str = self.command_str[:-1]

        else:
            self.command_str = command_str

    def __str__(self) -> str:
        return self.command_str

    def __repr__(self) -> str:
        return str(self)

class RecordingSession:
    command_list_file = "full_command_list.txt"
    max_command_len = 4
    def __init__(self, recording_instance=5) -> None:
        """
            recording_instance : how many times each unigram will be recorded in the dataset
        """
        self.recording_instance = recording_instance
        self.recorded_commands = []
        self.assembled_commands = []

        self.init_recorded_words_instances()
        self.init_parts()

        self.generate_command_to_record()
        

    def get_words_and_pos(self):
        words_and_pos = []
        with open(self.command_list_file, 'r') as f:
            for line in f.read().splitlines():
                words = line.split(' ')
                for i, word in enumerate(words):
                    if (word, i) not in words_and_pos:
                        words_and_pos.append((word, i))
        
        return words_and_pos

    def init_parts(self) -> None:
        self.parts = {}
        for ind in range(self.max_command_len):
            self.parts[ind] = []

    def init_recorded_words_instances(self) -> None:
        self.recorded_words_instances = {}

        for word, pos in self.get_words_and_pos():
            self.recorded_words_instances[(word, pos)] = 0

    def add_parts_from_command(self, command : Command) -> None:
        for part in command.parts:
            self.parts[part.ind].append(part)
    
    def add_recorded_words_instances(self, command : Command) -> None:
        for part in command.parts:
            self.recorded_words_instances[(part.transcription, part.ind)] += 1

    def next_command_to_record(self) -> Command:
        with open(self.command_list_file, 'r') as f:
            command_list = f.read().splitlines()

        scores = []
        for command_str in command_list:
            command = Command(command_str=command_str)
            score = 0
            
            for part in command.parts:
                score += self.recording_instance - self.recorded_words_instances[(part.transcription, part.ind)]
            
            scores.append(score)

        return Command(command_str=command_list[scores.index(max(scores))])

    def instance_criteria(self) -> bool:
        criteria = True
        for key in self.recorded_words_instances:
            if self.recorded_words_instances[key] < self.recording_instance:
                criteria = False
        
        return criteria

    def remove_useless_commands(self):
        commands_to_remove = []
        for command in self.recorded_commands:
            can_be_removed = True
            for part in command.parts:
                if not self.recorded_words_instances[(part.transcription, part.ind)] > self.recording_instance:
                    can_be_removed = False
            
            if can_be_removed:
                commands_to_remove.append(command)
                for part in command.parts:
                    self.recorded_words_instances[(part.transcription, part.ind)] -= 1

        
        for command in commands_to_remove:
            self.recorded_commands.remove(command)



    def generate_command_to_record(self): 
        #main method
        # get all the possible parts with the matching recording
        # generate all the possible combination of command with this parts
        # check in the full command list which one exist
        # assemble this new commands with if possibles parts from the same recording
        # if the recording_instance criteria is not met, record a new command and redo all this steps
        # /!\ the new command to record is chosen by the next_command_to_record method

        while not self.instance_criteria():

            new_command = self.next_command_to_record()
            self.recorded_commands.append(new_command)
            self.add_recorded_words_instances(new_command)

        self.remove_useless_commands()

        for command in self.recorded_commands:
            self.add_parts_from_command(command)
        
        print(len(self.recorded_commands))
        print(self.recorded_words_instances)

        #possible_commands = self.calculate_possible_command()
        #print(possible_commands)
    
    def calculate_possible_command(self):
        lists = []
        for ind in self.parts:
            lists.append(self.parts[ind])
    
        parts_combinations = list(itertools.product(*lists))
        possible_commands = []


        with open(self.command_list_file, 'r') as f:
            correct_commands = f.read().splitlines()

        for part_tuple in parts_combinations:
            command = Command(parts=part_tuple)
            if str(command) in correct_commands:
                possible_commands.append(command)

        return possible_commands





if __name__=='__main__':
    re_sess = RecordingSession(recording_instance=12)
