import json
import logging

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


annotation_file = "annotation.json"

nb_iteration_valid = 1
nb_iteration_test = 1

def prepare(data_folder, 
    save_json_train,
    save_json_valid,
    save_json_test, 
    kfold = 1, 
    skip_prep=False):


    with open(data_folder + annotation_file, 'r') as a:
        annotation_dict = json.load(a)
    a.close()

    iteration_dict = {}
    # Sort entries by command
    for entry in annotation_dict:
        full_command = annotation_dict[entry]["full_command"]
        if full_command not in iteration_dict:
            iteration_dict[full_command] = [entry]
        else:
            iteration_dict[full_command].append(entry)
        
    
    train_annotation_dict = {}
    valid_annotation_dict = {}
    test_annotation_dict = {}

    for full_command in iteration_dict:
        nb_iteration_total = len(iteration_dict[full_command])

        if nb_iteration_valid + nb_iteration_test >= nb_iteration_total:
            logger.info("Not enough iterations for '" + full_command + "'. Skipping it")

        else:
            for _ in range(nb_iteration_valid):
                entry = iteration_dict[full_command].pop()
                valid_annotation_dict[entry] = annotation_dict[entry]

            for _ in range(nb_iteration_test):
                entry = iteration_dict[full_command].pop()
                test_annotation_dict[entry] = annotation_dict[entry]
            
            while len(iteration_dict[full_command]) > 0:
                entry = iteration_dict[full_command].pop()
                train_annotation_dict[entry] = annotation_dict[entry]
    

    
    json.dump(train_annotation_dict, open(save_json_train, 'w'), indent=2)
    json.dump(valid_annotation_dict, open(save_json_valid, 'w'), indent=2)
    json.dump(test_annotation_dict, open(save_json_test, 'w'), indent=2)

    logger.info("Preparation finished.")