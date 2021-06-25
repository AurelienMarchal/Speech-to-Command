
import os
import sys
from wave import Wave_read
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)



class Speech2CommandBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        command_bos, _ = batch.command_encoded_bos

        if stage == sb.Stage.TRAIN and self.hparams.apply_data_augmentation:
            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, wav_lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                wavs_aug_tot.append(wavs_aug)

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            wav_lens = torch.cat([wav_lens] * self.n_augment)
            command_bos = torch.cat([command_bos] * self.n_augment)

        
        feats = self.modules.wav2vec2(wavs)

        encoder_out = self.hparams.enc(feats)

        embedded_dec_in = self.hparams.emb(command_bos)

        decoder_out, _ = self.hparams.dec(embedded_dec_in, encoder_out, wav_lens)
        
        logits = self.hparams.seq_lin(decoder_out)

        outputs = self.modules.softmax(logits)


        # Compute outputs
        #TO DO add beam searcher for VALID and TEST

        return outputs, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using command-id as label.
        """
        predictions_seq, lens = predictions
        uttid = batch.id
        
        command_eos, command_eos_lens = batch.command_encoded_eos
        commands, command_lens = batch.command_encoded
        
        

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and self.hparams.apply_data_augmentation:
            commands = torch.cat([commands] * self.n_augment, dim=0)
            command_lens = torch.cat([command_lens] * self.n_augment, dim=0)
            
            command_eos = torch.cat([command_eos] * self.n_augment, dim=0)
            command_eos_lens = torch.cat([command_eos_lens] * self.n_augment, dim=0)

        # compute the cost function
        loss_seq = self.hparams.seq_cost(predictions_seq, command_eos, length=command_eos_lens)
        loss= loss_seq

        if hasattr(self.hparams.lr_annealing_adam, "on_batch_end"):
            self.hparams.lr_annealing_adam.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions_seq, command_eos, length=command_eos_lens)

        if stage != sb.Stage.TRAIN:
            
            target_words = label_encoder.decode_torch(commands)
            predicted_words = label_encoder.decode_torch(torch.argmax(predictions_seq, dim=2))
            

            if stage== sb.Stage.TEST:
                print("\nTargets :", target_words)
                print("Predictions :", predicted_words)


        return loss


    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.seq_stats()


    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing_adam(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("opt", self.optimizer)

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]
    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.TextEncoder()
    #label_encoder.add_unk()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        #print(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("full_command", "cmd", "obj1", "prep", "obj2")
    @sb.utils.data_pipeline.provides(
        "full_command", 
        "command_encoded_list", 
        "command_encoded",
        "command_encoded_eos",
        "command_encoded_bos"
        )
    def text_pipeline(full_command, cmd, obj1, prep, obj2):
        yield full_command
        full_command_list = full_command.split(' ')
        command_encoded_list = label_encoder.encode_sequence(full_command_list)
        yield command_encoded_list
        command_encoded = torch.LongTensor(command_encoded_list)
        yield command_encoded
        command_encoded_eos = torch.LongTensor(label_encoder.append_eos_index(command_encoded_list))
        yield command_encoded_eos
        command_encoded_bos = torch.LongTensor(label_encoder.prepend_bos_index(command_encoded_list))
        yield command_encoded_bos


    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label":hparams["bos_index"],
        "eos_label":hparams["eos_index"]
    }

    label_encoder.load_or_create(
        path=lab_enc_file, 
        from_didatasets=[train_data], 
        output_key="full_command_list",
        special_labels=special_labels,
        sequence_input=True
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "command_encoded", "command_encoded_eos", "command_encoded_bos"]
    )

    return train_data, valid_data, test_data, label_encoder




if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    from prepare import prepare

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "kfold": hparams["k_fold"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)
    


    # Brain class initialization
    speaker_brain = Speech2CommandBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        checkpointer=hparams["checkpointer"],
    )


    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Load the best checkpoint for evaluation
    test_stats = speaker_brain.evaluate(
        test_set=test_data,
        min_key="ErrorRate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )



# Test modification
