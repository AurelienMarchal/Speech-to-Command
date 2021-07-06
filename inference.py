from ast import dump
import speechbrain as sb
import torch
import sys
import os
import json
import glob
from hyperpyyaml import load_hyperpyyaml
from train import Speech2CommandBrain


def create_json(hparams):
    json_path = hparams["json_path"]
    inf_folder = hparams["inference_folder"]

    annotation = {}

    for filename in glob.glob(os.path.join(inf_folder, '*.wav')):
        annotation[os.path.splitext(filename)[0].split('/')[-1]] = {"wav": filename}
    
    with open(json_path, 'w') as json_file:
        json.dump(annotation, json_file, indent=2)
    
    json_file.close()

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    create_json(hparams)

    # 1. Declarations:
    data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["json_path"]
    )

    datasets = [data]
    
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder = sb.dataio.encoder.TextEncoder.from_saved(lab_enc_file)


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes()
    @sb.utils.data_pipeline.provides("command_encoded_bos")
    def text_pipeline():
        command_encoded_bos = torch.LongTensor(label_encoder.get_bos_index())
        return command_encoded_bos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "command_encoded_bos"]
    )

    return data, label_encoder


if __name__ == '__main__':
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create dataset from wav
    data, label_encoder = dataio_prep(hparams)


    # Brain class initialization
    speaker_brain = Speech2CommandBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    transcribes = speaker_brain.transcribe_dataset(
        dataset=data, 
        min_key="ErrorRate", 
        loader_kwargs=hparams["test_dataloader_opts"]
    )

    print(transcribes)
