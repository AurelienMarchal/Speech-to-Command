import find_voiced_segment
import glob

import pyaudio
import wave
from scipy.io import wavfile

import speechbrain as sb
import torch
import sys
import os
import json
import glob
import logging
from hyperpyyaml import load_hyperpyyaml
from train import Speech2CommandBrain
logger = logging.getLogger(__name__)

def create_json(hparams):
    json_path = hparams["json_path"]
    inf_folder = hparams["inference_folder"]

    annotation = {}

    for filename in glob.glob(os.path.join(inf_folder, '*.wav')):
        annotation[os.path.splitext(filename)[0].split('/')[-1]] = {"wav": filename}
    
    with open(json_path, 'w') as json_file:
        json.dump(annotation, json_file, indent=2)
    
    json_file.close()

    return len(annotation)

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    nb_files = create_json(hparams)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder = sb.dataio.encoder.TextEncoder.from_saved(lab_enc_file)

    if nb_files == 0:
        logger.warning("At least one file must be in " + hparams["inference_folder"])
        return None, label_encoder

    # 1. Declarations:
    data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["json_path"]
    )

    datasets = [data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig"]
    )

    return data, label_encoder


if __name__ == '__main__':
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    logger.info("Loading model")

    # Brain class initialization
    speaker_brain = Speech2CommandBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    filename = hparams["recording_file"]

    while True:

        inp = input("Do you wish to record ? (press 'q' and enter to quit)\n")

        if inp == 'q':
            break

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        logger.info("Recording")
        stream = p.open(format=hparams["sample_format"],
                        channels=hparams["channels"],
                        rate=hparams["sample_rate"],
                        frames_per_buffer=hparams["chunk"],
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for x seconds
        for i in range(0, int(hparams["sample_rate"] / hparams["chunk"] * hparams["seconds"])):
            data = stream.read(hparams["chunk"])
            frames.append(data)

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        logger.info("Finished recording")

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(hparams["channels"])
        wf.setsampwidth(p.get_sample_size(hparams["sample_format"]))
        wf.setframerate(hparams["sample_rate"])
        wf.writeframes(b''.join(frames))
        wf.close()

        # cut useless segment
        #find_voiced_segment.main(3, filename)

        # Create dataset from wav
        data, label_encoder = dataio_prep(hparams)
        speaker_brain.label_encoder = label_encoder

        if data is not None:
            transcribes = speaker_brain.transcribe_dataset(
                dataset=data, 
                min_key="ErrorRate", 
                loader_kwargs=hparams["test_dataloader_opts"]
            )

            print("Predictions :" + str(transcribes))
            logger.info("Predictions :" + str(transcribes))
