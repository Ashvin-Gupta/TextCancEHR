import os
import pickle
import random
from logging import Logger
import pandas as pd
import torch
from tqdm import tqdm


class RolloutEvaluationDataset(torch.utils.data.Dataset):
    """
    Dataset for running evaluation of a Nightingale model. This class takes a directory (data_dir) of pickled tokenized data
    produced by the ehr-tokenization pipeline. Each pickle file is loaded and a rollout datapoint is attempted to be generated for each subject.
    This dataset is used for running evaluations where you want to see the models ability to predict an end token given a context window.

    Args:
        dataset_dir (str): The directory containing the pickled tokenized data.
        vocab_path (str): The path to the vocabulary file.
        sequence_length (int): The length of the input and target token sequences.
        start_token_id (int, optional): The id of the start token. Either this or start_token_str must be provided.
        end_token_ids (list[int], optional): The ids of the end tokens. Either this or end_token_strs must be provided.
        start_token_str (str, optional): The string representation of the start token. Either this or start_token_id must be provided.
        end_token_strs (list[str], optional): The string representations of the end tokens. Either this or end_token_ids must be provided.
        include_patients_without_end_token (bool, optional): Whether to include patients who have no valid end token after the start token.
            If True, these patients will be included with end_token=-1 and end_token_idx=-1.
            If False (default), these patients will be excluded from the dataset.
        seconds_offset (int, optional): If this is not None then seconds_offset will be used to expand the input window size of the rollout.
            Tokens within start_token.timestamp + seconds_offset will be included in the input window.
            Tokens at the start of the context window will be dropped to make room for the seconds_offset tokens, so the length of the input token sequence is sequence_length.
        logger (Logger, optional): The logger to use for logging dataset statistics and warnings.
    """

    def __init__(
        self,
        dataset_dir: str,
        vocab_path: str,
        sequence_length: int,
        insert_static_demographic_tokens: bool = True,
        start_token_id: int = None,
        end_token_ids: list[int] = None,
        start_token_str: str = None,
        end_token_strs: list[str] = None,
        include_patients_without_end_token: bool = False,
        seconds_offset: int = None,
        logger: Logger = None
        ) -> None:

        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.insert_static_demographic_tokens = insert_static_demographic_tokens
        self.include_patients_without_end_token = include_patients_without_end_token
        self.seconds_offset = seconds_offset
        self.logger = logger
        self.vocab = pd.read_csv(vocab_path) # columns are token, str, count

        # set start token
        if start_token_id is not None:
            self.start_token_id = start_token_id
            start_token_matches = self.vocab[self.vocab["token"] == start_token_id]["str"]
            if len(start_token_matches) == 0:
                raise ValueError(f"start_token_id {start_token_id} not found in vocab")
            self.start_token_str = start_token_matches.values[0]
        elif start_token_str is not None:
            self.start_token_str = start_token_str
            start_token_matches = self.vocab[self.vocab["str"] == start_token_str]["token"]
            if len(start_token_matches) == 0:
                raise ValueError(f"start_token_str '{start_token_str}' not found in vocab")
            self.start_token_id = start_token_matches.values[0]
        else:
            raise ValueError("Either start_token_id or start_token_str must be provided")
        
        # set end tokens
        if end_token_ids is not None:
            self.end_token_ids = end_token_ids
            self.end_token_strs = []
            for end_token_id in end_token_ids:
                end_token_matches = self.vocab[self.vocab["token"] == end_token_id]["str"]
                if len(end_token_matches) == 0:
                    raise ValueError(f"end_token_id {end_token_id} not found in vocab")
                self.end_token_strs.append(end_token_matches.values[0])
        elif end_token_strs is not None:
            self.end_token_strs = end_token_strs
            self.end_token_ids = []
            for end_token_str in end_token_strs:
                end_token_matches = self.vocab[self.vocab["str"] == end_token_str]["token"]
                if len(end_token_matches) == 0:
                    raise ValueError(f"end_token_str '{end_token_str}' not found in vocab")
                self.end_token_ids.append(end_token_matches.values[0])
        else:
            raise ValueError("Either end_token_ids or end_token_strs must be provided")

        self.data = self._load_data_from_dir(dataset_dir)
        if len(self.data) == 0:
            raise ValueError("No matching datapoints found in dataset, check dataset directory and start and end tokens")

    def _load_data_from_dir(self, dataset_dir: str) -> list:
        """
        Loads data from the data directory and populates self.data.

        Args:
            data_dir (str): The directory containing the pickled tokenized data.

        Returns:
            data (list): A list of dictionaries containing the data for each sample.
        """

        data = []
        
        # Statistics for logging
        total_patients = 0
        patients_with_valid_rollout = 0
        patients_without_end_token = 0
        patients_filtered_other_reasons = 0

        # get all the pickle files in the dataset directory
        file_paths = [
            os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.endswith(".pkl")
        ]

        for file_path in tqdm(file_paths, desc="Loading data"):
            with open(file_path, "rb") as f:
                for subject_data in pickle.load(f):
                    total_patients += 1
                    
                    # try to generate a rollout datapoint
                    try:
                        input_tokens, input_timestamps, end_token, start_token_idx, end_token_idx = self._generate_rollout(subject_data["tokens"], subject_data["timestamps"])
                        
                        # map to tensors
                        input_tokens = torch.tensor(input_tokens)
                        input_timestamps = torch.tensor(input_timestamps)
                        end_token = torch.tensor(end_token)
                        start_token_idx = torch.tensor(start_token_idx)
                        end_token_idx = torch.tensor(end_token_idx)

                        # Check if patient has a valid end token
                        if end_token == -1:
                            patients_without_end_token += 1
                            if not self.include_patients_without_end_token:
                                if self.logger is not None:
                                    self.logger.debug(f"Excluding subject {subject_data['subject_id']} - no valid end token found")
                                continue
                        
                        # insert static demographic tokens
                        if self.insert_static_demographic_tokens:
                            # Convert to tensors to handle numpy arrays properly
                            tokens_tensor = torch.tensor(subject_data["tokens"])
                            timestamps_tensor = torch.tensor(subject_data["timestamps"])
                            static_demographic_token_ids = tokens_tensor[timestamps_tensor == 0]
                            
                            # Adjust input sequence to maintain total sequence_length
                            static_token_count = len(static_demographic_token_ids)
                            if static_token_count > 0:
                                # Truncate input_tokens to make room for static tokens
                                max_input_tokens = self.sequence_length - static_token_count
                                if len(input_tokens) > max_input_tokens:
                                    input_tokens = input_tokens[-max_input_tokens:]
                                    input_timestamps = input_timestamps[-max_input_tokens:]
                                
                                # Prepend static demographic tokens
                                input_tokens = torch.cat([static_demographic_token_ids, input_tokens])
                                input_timestamps = torch.cat([torch.zeros(static_token_count, dtype=input_timestamps.dtype), input_timestamps])

                        # Include the patient
                        patients_with_valid_rollout += 1
                        data.append({
                            "subject_id": torch.tensor(subject_data["subject_id"]),
                            "input_tokens": input_tokens,
                            "input_timestamps": input_timestamps,
                            "end_token": end_token,
                            "start_token_idx": start_token_idx,
                            "end_token_idx": end_token_idx
                        })
                        
                    except ValueError as e:
                        patients_filtered_other_reasons += 1
                        if self.logger is not None:
                            self.logger.debug(f"Excluding subject {subject_data['subject_id']} - {str(e)}")
                        continue

        # Log key dataset statistics
        if self.logger is not None:
            self.logger.info(f"Dataset loading complete from {dataset_dir}")
            self.logger.info(f"Total patients processed: {total_patients}")
            self.logger.info(f"Patients included in dataset: {len(data)} ({len(data)/total_patients*100:.1f}%)")
            self.logger.info(f"Patients without end tokens: {patients_without_end_token} ({'included' if self.include_patients_without_end_token else 'excluded'})")
            
        return data
    
    def _generate_rollout(self, tokens: list[int], timestamps: list[int]) -> list[int]:
        """
        Generates a rollout datapoint that starts from a start token and ends with an end token.

        Args:
            tokens (list[int]): The list of tokens to generate a rollout from.
            timestamps (list[int]): The list of timestamps to generate a rollout from.

        Returns:
            input_tokens (list[int]): The list of tokens to generate a rollout from.
            input_timestamps (list[int]): The list of timestamps to generate a rollout from.
            end_token (int): The end token that the rollout ends with.
            end_token_idx (int): The index of the end token that the rollout ends with.
        """

        # find the index of the last instance of the start token in the tokens list
        try:
            start_token_idx = len(tokens) - 1 - tokens[::-1].index(self.start_token_id)
        except ValueError:
            if self.logger is not None:
                self.logger.warning(f"Start token {self.start_token_id} not found in tokens")
            raise ValueError(f"Start token {self.start_token_id} not found in tokens")
        
        # if there are less than sequence_length tokens up to and including the start token,
        # raise an error as there is not enough tokens for the context window
        # TODO: in future this should use padding and a mask so that these sample can be used for evaluation
        if start_token_idx + 1 < self.sequence_length:
            if self.logger is not None:
                self.logger.warning(f"Less than {self.sequence_length} tokens before start token")
            raise ValueError(f"Less than {self.sequence_length} tokens before start token")
        
        # find the first instance of any of the end token ids that occur after the start token index
        for steps, token_id in enumerate(tokens[start_token_idx + 1:]):
            if token_id in self.end_token_ids:
                end_token_idx = steps + start_token_idx + 1
                end_token = token_id
                break
        else:
            # if no end token is found then return end_token = -1 and end_token_idx = -1
            end_token = -1
            end_token_idx = -1
        
        # create the input sequence
        if self.seconds_offset is None:
            # Default behavior: context window ends with the start token
            input_sequence_start_idx = start_token_idx - self.sequence_length + 1
            input_tokens = tokens[input_sequence_start_idx:start_token_idx + 1]
            input_timestamps = timestamps[input_sequence_start_idx:start_token_idx + 1]
        else:
            # Time-based sequence creation
            start_timestamp = timestamps[start_token_idx]
            
            if self.seconds_offset > 0:
                # Forward-looking: include tokens after start_token within time window
                end_timestamp = start_timestamp + self.seconds_offset
                sequence_end_idx = start_token_idx
                for i in range(start_token_idx + 1, len(tokens)):
                    if timestamps[i] <= end_timestamp:
                        sequence_end_idx = i
                    else:
                        break
                
                # Get time window tokens
                time_tokens = tokens[start_token_idx:sequence_end_idx + 1]
                time_timestamps = timestamps[start_token_idx:sequence_end_idx + 1]
                
            else:
                # Backward-looking: include tokens before start_token within time window
                lookback_timestamp = start_timestamp + self.seconds_offset  # seconds_offset is negative
                sequence_start_idx = start_token_idx
                for i in range(start_token_idx - 1, -1, -1):
                    if timestamps[i] >= lookback_timestamp:
                        sequence_start_idx = i
                    else:
                        break
                
                # Get time window tokens (excluding start_token for backward-looking)
                time_tokens = tokens[sequence_start_idx:start_token_idx]
                time_timestamps = timestamps[sequence_start_idx:start_token_idx]
            
            # Ensure sequence_length constraint
            if len(time_tokens) > self.sequence_length:
                # Keep most recent tokens, warn if start_token dropped
                if self.seconds_offset > 0 and self.logger:
                    self.logger.warning(f"Time window exceeds sequence_length, start_token may be dropped")
                input_tokens = time_tokens[-self.sequence_length:]
                input_timestamps = time_timestamps[-self.sequence_length:]
            elif len(time_tokens) < self.sequence_length:
                # Pad with earlier context
                tokens_needed = self.sequence_length - len(time_tokens)
                pad_start_idx = max(0, (sequence_start_idx if self.seconds_offset < 0 else start_token_idx) - tokens_needed)
                pad_tokens = tokens[pad_start_idx:sequence_start_idx if self.seconds_offset < 0 else start_token_idx]
                input_tokens = pad_tokens + time_tokens
                input_timestamps = timestamps[pad_start_idx:sequence_start_idx if self.seconds_offset < 0 else start_token_idx] + time_timestamps
            else:
                input_tokens = time_tokens
                input_timestamps = time_timestamps

        return input_tokens, input_timestamps, end_token, start_token_idx, end_token_idx
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]
    

if __name__ == "__main__":

    dataset_parent_dir = "/home/joshua/data/mimic/mimic_iv/meds/mimic_iv_meds/tokenized_data/ethos_timetokens"
    dataset_dir = os.path.join(dataset_parent_dir, "tuning")
    vocab_path = os.path.join(dataset_parent_dir, "vocab.csv")

    start_token_str = "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"

    end_token_strs = ["MEDS_DEATH", "TRANSFER_TO//discharge//UNKNOWN", "HOSPITAL_DISCHARGE//HOME", "HOSPITAL_DISCHARGE//UNK"]

    dataset = RolloutEvaluationDataset(dataset_dir, vocab_path, sequence_length=128, start_token_str=start_token_str, end_token_strs=end_token_strs, logger=None)

    print(dataset.start_token_id)
    print(dataset.start_token_str)
    print(dataset.end_token_ids)
    print(dataset.end_token_strs)

    for x in dataset:
        print(x['input_tokens'])
        print(x['end_token'])
        print(dataset.vocab[dataset.vocab["token"] == int(x['end_token'])]["str"].values[0])
        print(x['end_token_idx'])
        input()