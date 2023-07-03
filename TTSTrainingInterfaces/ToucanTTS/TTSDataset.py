import os
import statistics

import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Aligner.Aligner import Aligner
from Aligner.CodecAlignerDataset import CodecAlignerDataset
from Preprocessing.CodecAudioPreprocessor import CodecAudioPreprocessor
from Preprocessing.TextFrontend import get_language_id
from Preprocessing.articulatory_features import get_feature_to_index_lookup
from TTSTrainingInterfaces.ToucanTTS.DurationCalculator import DurationCalculator
from TTSTrainingInterfaces.ToucanTTS.EnergyCalculator import EnergyCalculator
from TTSTrainingInterfaces.ToucanTTS.PitchCalculator import Parselmouth


class TTSDataset(Dataset):

    def __init__(self,
                 path_to_transcript_dict,
                 acoustic_checkpoint_path,
                 cache_dir,
                 lang,
                 loading_processes=os.cpu_count() if os.cpu_count() is not None else 30,
                 min_len_in_seconds=1,
                 max_len_in_seconds=15,
                 reduction_factor=1,
                 device=torch.device("cpu"),
                 rebuild_cache=False,
                 ctc_selection=True,
                 save_imgs=False):
        self.codec_wrapper = CodecAudioPreprocessor(input_sr=-1)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cache_dir, "tts_train_cache.pt")) or rebuild_cache:
            if not os.path.exists(os.path.join(cache_dir, "aligner_train_cache.pt")) or rebuild_cache:
                CodecAlignerDataset(path_to_transcript_dict=path_to_transcript_dict,
                                    cache_dir=cache_dir,
                                    lang=lang,
                                    loading_processes=loading_processes,
                                    min_len_in_seconds=min_len_in_seconds,
                                    max_len_in_seconds=max_len_in_seconds,
                                    rebuild_cache=rebuild_cache,
                                    device=device)
            datapoints = torch.load(os.path.join(cache_dir, "aligner_train_cache.pt"), map_location='cpu')
            # we use the aligner dataset as basis and augment it to contain the additional information we need for tts.
            dataset = datapoints[0]
            # index 1 is deprecated
            # index 2 are the speaker embeddings used for the reconstruction loss of the Aligner, we don't need them anymore
            filepaths = datapoints[3]

            print("... building dataset cache ...")
            self.datapoints = list()
            self.ctc_losses = list()

            acoustic_model = Aligner()
            acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"])

            # ==========================================
            # actual creation of datapoints starts here
            # ==========================================

            acoustic_model = acoustic_model.to(device)
            _, _orig_sr = sf.read(filepaths[0])
            parsel = Parselmouth(reduction_factor=reduction_factor, fs=_orig_sr)
            energy_calc = EnergyCalculator(reduction_factor=reduction_factor, fs=_orig_sr)
            dc = DurationCalculator(reduction_factor=reduction_factor)
            vis_dir = os.path.join(cache_dir, "duration_vis")
            os.makedirs(vis_dir, exist_ok=True)

            for index in tqdm(range(len(dataset))):
                filepath = filepaths[index]
                raw_wave, sr = sf.read(filepath)
                if _orig_sr != sr:
                    print(f"Not all files have the same sampling rate! Please fix and re-run.  -- triggered by {filepath}")

                norm_wave_length = torch.LongTensor([len(raw_wave)])

                text = dataset[index][0]
                melspec = self.codec_wrapper.indexes_to_codec_frames(dataset[index][1].int().transpose(0, 1)).transpose(0, 1).detach()
                melspec_length = torch.LongTensor([len(dataset[index][1])])

                # We deal with the word boundaries by having 2 versions of text: with and without word boundaries.
                # We note the index of word boundaries and insert durations of 0 afterwards
                text_without_word_boundaries = list()
                indexes_of_word_boundaries = list()
                for phoneme_index, vector in enumerate(text):
                    if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                        text_without_word_boundaries.append(vector.numpy().tolist())
                    else:
                        indexes_of_word_boundaries.append(phoneme_index)
                matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)

                alignment_path, ctc_loss = acoustic_model.inference(mel=melspec.to(device),
                                                                    tokens=matrix_without_word_boundaries.to(device),
                                                                    save_img_for_debug=os.path.join(vis_dir, f"{index}.png") if save_imgs else None,
                                                                    return_ctc=True)

                cached_duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

                for index_of_word_boundary in indexes_of_word_boundaries:
                    cached_duration = torch.cat([cached_duration[:index_of_word_boundary],
                                                 torch.LongTensor([0]),  # insert a 0 duration wherever there is a word boundary
                                                 cached_duration[index_of_word_boundary:]])

                last_vec = None
                for phoneme_index, vec in enumerate(text):
                    if last_vec is not None:
                        if last_vec.numpy().tolist() == vec.numpy().tolist():
                            # we found a case of repeating phonemes!
                            # now we must repair their durations by giving the first one 3/5 of their sum and the second one 2/5 (i.e. the rest)
                            dur_1 = cached_duration[phoneme_index - 1]
                            dur_2 = cached_duration[phoneme_index]
                            total_dur = dur_1 + dur_2
                            new_dur_1 = int((total_dur / 5) * 3)
                            new_dur_2 = total_dur - new_dur_1
                            cached_duration[phoneme_index - 1] = new_dur_1
                            cached_duration[phoneme_index] = new_dur_2
                    last_vec = vec

                cached_energy = energy_calc(input_waves=torch.tensor(raw_wave).unsqueeze(0),
                                            input_waves_lengths=norm_wave_length,
                                            feats_lengths=melspec_length,
                                            text=text,
                                            durations=cached_duration.unsqueeze(0),
                                            durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu()

                cached_pitch = parsel(input_waves=torch.tensor(raw_wave).unsqueeze(0),
                                      input_waves_lengths=norm_wave_length,
                                      feats_lengths=melspec_length,
                                      text=text,
                                      durations=cached_duration.unsqueeze(0),
                                      durations_lengths=torch.LongTensor([len(cached_duration)]))[0].squeeze(0).cpu()

                self.datapoints.append([dataset[index][0],  # text tensor
                                        torch.LongTensor([len(dataset[index][0])]),  # length of text tensor
                                        dataset[index][1],  # codec tensor (in index form)
                                        torch.LongTensor([len(dataset[index][1])]),  # length of codec tensor
                                        cached_duration.cpu(),  # duration
                                        cached_energy,  # energy
                                        cached_pitch,  # pitch
                                        None,  # deprecated,
                                        filepaths[index]  # path to the associated original raw audio file
                                        ])
                self.ctc_losses.append(ctc_loss)

            # =============================
            # done with datapoint creation
            # =============================

            if ctc_selection and len(self.datapoints) > 300:  # for less than 300 datapoints, we should not throw away anything.
                # now we can filter out some bad datapoints based on the CTC scores we collected
                mean_ctc = sum(self.ctc_losses) / len(self.ctc_losses)
                std_dev = statistics.stdev(self.ctc_losses)
                threshold = mean_ctc + (std_dev * 3.5)
                for index in range(len(self.ctc_losses), 0, -1):
                    if self.ctc_losses[index - 1] > threshold:
                        self.datapoints.pop(index - 1)
                        print(f"Removing datapoint {index - 1}, because the CTC loss is 3.5 standard deviations higher than the mean. \n ctc: {round(self.ctc_losses[index - 1], 4)} vs. mean: {round(mean_ctc, 4)}")

            # save to cache
            if len(self.datapoints) > 0:
                torch.save(self.datapoints, os.path.join(cache_dir, "tts_train_cache.pt"))
            else:
                import sys
                print("No datapoints were prepared! Exiting...")
                sys.exit()
        else:
            # just load the datapoints from cache
            self.datapoints = torch.load(os.path.join(cache_dir, "tts_train_cache.pt"), map_location='cpu')

        self.cache_dir = cache_dir
        self.language_id = get_language_id(lang)
        print(f"Prepared a TTS dataset with {len(self.datapoints)} datapoints in {cache_dir}.")

    def __getitem__(self, index):
        return self.datapoints[index][0], \
               self.datapoints[index][1], \
               self.codec_wrapper.indexes_to_codec_frames(self.datapoints[index][2].int().transpose(0, 1)).transpose(0, 1).detach(), \
               self.datapoints[index][3], \
               self.datapoints[index][4], \
               self.datapoints[index][5], \
               self.datapoints[index][6], \
               self.datapoints[index][7], \
               self.language_id

    def __len__(self):
        return len(self.datapoints)

    def remove_samples(self, list_of_samples_to_remove):
        for remove_id in sorted(list_of_samples_to_remove, reverse=True):
            self.datapoints.pop(remove_id)
        torch.save(self.datapoints, os.path.join(self.cache_dir, "tts_train_cache.pt"))
        print("Dataset updated!")