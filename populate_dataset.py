import json
import string
import random
import os

import find_voiced_segment

import pyaudio
import wave
import contextlib


annotation_file_json = "dataset/annotation.json"
full_command_list_file = "full_command_list.txt"
data_dict = "./dataset/wavs/"

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000 # Record at 16000 samples per second
seconds = 5
# Create an interface to PortAudio
p = pyaudio.PyAudio()

# nb of times a command needs to be recorded
nb_iterations = 5

speaker = 'aurelien_marchal'

def id_generator(size=10, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generate_unique_entry_id(annotation_dict, full_command):
    full_command = full_command.replace(' ', '_')
    id_ = id_generator()
    while full_command + '_' + id_ in annotation_dict:
        id_ = id_generator()
    return full_command + '_' + id_




def count_full_command_iterations(annotation_dict, speaker):
    full_command_iterations = {}
    with open(full_command_list_file) as f:
        full_command_list = f.read().splitlines()
        for full_command in full_command_list:
            full_command_iterations[full_command] = 0
            for entry in annotation_dict:
                if annotation_dict[entry]['full_command'] == full_command and annotation_dict[entry]['speaker'] == speaker:
                    full_command_iterations[full_command] += 1
    f.close()
    return full_command_iterations

def write_entry(annotation_dict, full_path, full_command, speaker, entry_id, duration):

    entry = {
        "full_command":full_command,
        "wav":full_path,
        "duration": duration,
        "speaker":speaker,
        "cmd": full_command.split(' ')[0],
        "obj1":full_command.split(' ')[1],
        "prep":full_command.split(' ')[2],
        "obj2":full_command.split(' ')[3]
    }

    print('Writing entry ', entry_id, ':' , entry)

    annotation_dict[entry_id] = entry


def record_entry(annotation_dict, full_command, speaker):

    entry_id =  generate_unique_entry_id(annotation_dict, full_command) 
    full_path = data_dict + speaker + '/' + entry_id + '.wav'
    print('------------> The command is : [', full_command, '] <------------')
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for x seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    

    print('Finished recording')

    print('Saving at : ', full_path)

    # Save the recorded data as a WAV file
    wf = wave.open(full_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    # cut useless segment
    num_segments = find_voiced_segment.main(2, full_path)

    if num_segments > 1:
        print("!!! Bad recording. Found more than one voiced segment, restart recording")
        os.remove(full_path)
        record_entry(annotation_dict, full_command, speaker)
        return 0
    

    #get final duration

    with contextlib.closing(wave.open(full_path,'r')) as f:
        frames_ = f.getnframes()
        rate_ = f.getframerate()
        duration = frames_ / float(rate_)


    write_entry(annotation_dict, full_path, full_command, speaker, entry_id, duration)



if __name__ == '__main__':

    

    with open(annotation_file_json, 'r') as a:
        annotation_dict = json.load(a)
        
        full_command_iterations = count_full_command_iterations(annotation_dict, speaker)
        print(full_command_iterations)
        
        a.close()
        for full_command in full_command_iterations:
            for i in range(full_command_iterations[full_command], nb_iterations):

                print(f'{i+1}/{nb_iterations}')
                # start recording session
                record_entry(annotation_dict, full_command, speaker)

                #Writing in annotation.json
                with open(annotation_file_json, 'w') as a:
                    json.dump(annotation_dict, a)
                    
                a.close()

p.terminate()