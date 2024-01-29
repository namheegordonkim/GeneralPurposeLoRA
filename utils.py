import os
import struct
from array import array
from typing import Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainingArguments, TrainerState, TrainerControl, GenerationConfig
from transformers.integrations import TensorBoardCallback

from my_tokenizers import RVQTokenizer


class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))


def get_mnist_data(mnist_dir):
    training_images_filepath = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(mnist_dir, 'train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    return x_test, x_train, y_test, y_train


class EmpiricalEvalCallback(TensorBoardCallback):

    def __init__(self, eval_every: int):
        super().__init__()
        self.eval_every = eval_every

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with torch.no_grad():
            kwargs['model'].eval()
            print("Evaluating token accuracy, true samples")
            for token_offset in np.arange(0, 1):
                # Training empirical error
                all_generated = []
                all_answers = []
                generation_config = GenerationConfig(pad_token_id=0, eos_token_id=1, bos_token_id=0, max_new_tokens=1)
                for x in kwargs['train_dataloader']:
                    input_ids = x['input_ids'][:16]
                    labels = x['labels'][:16]

                    tiled_indices = torch.repeat_interleave(torch.arange(labels.shape[-1])[None], labels.shape[0], 0).to(input_ids.device)
                    question_lengths = torch.sum(labels == -100, dim=-1) + token_offset
                    yes = tiled_indices < question_lengths[:, None]
                    questions = torch.zeros_like(input_ids)
                    questions[torch.as_tensor(yes.detach().cpu().numpy()[..., ::-1].copy())] = input_ids[yes]

                    yes = tiled_indices >= question_lengths[:, None]
                    answer_lengths = torch.sum(yes, dim=-1)
                    receiver_mask = tiled_indices < answer_lengths[:, None]
                    answers = torch.zeros_like(input_ids)
                    answers[receiver_mask] = input_ids[yes]

                    generated = kwargs['model'].generate(inputs=questions, generation_config=generation_config)

                    all_generated.append(generated[:, [-1]])
                    all_answers.append(answers[:, [0]])
                    break

                all_generated = torch.cat(all_generated, dim=0)
                all_answers = torch.cat(all_answers, dim=0)

                token_empirical_acc = torch.mean((all_generated[:, 0] == all_answers[:, 0]).float())
                print(f"Empirical Token {token_offset} Accuracy (Train): {token_empirical_acc.data:.4f}")
                self.tb_writer.add_scalar(f"train/token_{token_offset}_acc", token_empirical_acc.data, state.global_step)

                all_generated = []
                all_answers = []
                for x in kwargs['eval_dataloader']:
                    input_ids = x['input_ids']
                    labels = x['labels']

                    tiled_indices = torch.repeat_interleave(torch.arange(labels.shape[-1])[None], labels.shape[0], 0).to(input_ids.device)
                    question_lengths = torch.sum(labels == -100, dim=-1) + token_offset
                    yes = tiled_indices < question_lengths[:, None]
                    questions = torch.zeros_like(input_ids)
                    questions[torch.as_tensor(yes.detach().cpu().numpy()[..., ::-1].copy())] = input_ids[yes]

                    yes = tiled_indices >= question_lengths[:, None]
                    answer_lengths = torch.sum(yes, dim=-1)
                    receiver_mask = tiled_indices < answer_lengths[:, None]
                    answers = torch.zeros_like(input_ids)
                    answers[receiver_mask] = input_ids[yes]

                    generated = kwargs['model'].generate(inputs=questions, generation_config=generation_config)

                    all_generated.append(generated[:, [-1]])
                    all_answers.append(answers[:, [0]])
                    # break

                all_generated = torch.cat(all_generated, dim=0)
                all_answers = torch.cat(all_answers, dim=0)

                token_empirical_acc = torch.mean((all_generated[:, 0] == all_answers[:, 0]).float())
                print(f"Empirical Token {token_offset} Accuracy (Eval): {token_empirical_acc.data:.4f}")
                self.tb_writer.add_scalar(f"eval/token_{token_offset}_acc", token_empirical_acc.data, state.global_step)

            kwargs['model'].train()


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        return np.clip((arr - self.mean) / np.sqrt(self.var + self.epsilon), -1000, 1000)

    def unnormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * np.sqrt(self.var + self.epsilon) + self.mean


class MNISTDatasetMaker:
    """
    Given Numpy arrays, prepares HuggingFace training / validation data
    """

    def __init__(self):
        # To be dynamically populated during `setup()`
        self.x_tokenizer = None
        self.x_scaler = None

        self.x_vocab_size = 2000
        self.x_code_length = 8

    def setup(self, setup_x: np.ndarray):
        num_patches = 4  # You can change this to the desired number of patches
        n_batches = setup_x.shape[0]
        patch_size = 28 // int(num_patches ** 0.5)
        n_idxs_per_axis = 28 // patch_size
        x_patches = np.empty((n_batches, num_patches, patch_size, patch_size))

        # Extract patches
        patch_idx = 0
        for i in range(n_idxs_per_axis):
            for j in range(n_idxs_per_axis):
                patch = setup_x[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                x_patches[:, patch_idx] = patch
                patch_idx += 1

        x_patches_tensor = torch.as_tensor(x_patches, dtype=torch.float)
        x_patches_tensor = x_patches_tensor / 255 * 2 - 1  # input normalization (256 possible greyscale pixel values), scaled to [-1, 1]
        self.x_tokenizer = RVQTokenizer(patch_size ** 2, self.x_code_length, self.x_vocab_size, 50, False, False)
        self.x_tokenizer.build_codebook(x_patches_tensor.reshape(-1, patch_size ** 2), batch_size=1000, device="cuda")

    def make(self, source_x: np.ndarray, source_y: np.ndarray):
        num_patches = 4  # You can change this to the desired number of patches
        n_batches = source_x.shape[0]
        patch_size = 28 // int(num_patches ** 0.5)
        n_idxs_per_axis = 28 // patch_size
        x_patches = np.empty((n_batches, num_patches, patch_size, patch_size))

        # Extract patches
        patch_idx = 0
        for i in range(n_idxs_per_axis):
            for j in range(n_idxs_per_axis):
                patch = source_x[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                x_patches[:, patch_idx] = patch
                patch_idx += 1

        x_patches_tensor = torch.as_tensor(x_patches, dtype=torch.float)
        x_patches_tensor = x_patches_tensor / 255 * 2 - 1

        x_patches_encoded, x_patches_quantized = self.x_tokenizer.encode(x_patches_tensor.reshape(-1, patch_size ** 2), device="cpu")
        x_patches_encoded = x_patches_encoded.reshape(n_batches, -1)
        x_patches_quantized = x_patches_quantized.reshape(n_batches, num_patches, patch_size, patch_size)

        # Get quantization error
        q_deltas = torch.abs(x_patches_quantized.detach().cpu() - x_patches_tensor).detach().cpu().numpy()
        q_mean = q_deltas.mean()
        q_std = q_deltas.std()
        q_max = q_deltas.max()

        print(f"Mean x quantization error : {q_mean:.3f}")
        print(f"Std x quantization error : {q_std:.3f}")
        print(f"Max x quantization error : {q_max:.3f}")

        # + 12 and + 2 because:
        # 12 total output vocabulary; 0 for padding, 1 for <EOS>, and 2-11 for the 10 digits (0-9)
        # To make the input vocabulary non-overlapping with the output vocabulary, we start input vocabulary at 12
        input_ids_np = np.concatenate([
            x_patches_encoded.cpu().detach().numpy().reshape(x_patches_encoded.shape[0], -1) + 12,
            source_y[:, None] + 2,
            np.ones((x_patches_encoded.shape[0], 1), dtype=int),
        ], axis=-1)
        labels_np = input_ids_np * 1
        labels_np[:, :-2] = -100  # "unused" for HuggingFace Llama
        attention_mask_np = np.ones_like(input_ids_np)

        input_ids = input_ids_np.tolist()
        labels = labels_np.tolist()
        attention_mask = attention_mask_np.tolist()

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        dataset = Dataset.from_dict(data_dict)
        return dataset


class ThroughDataset(Dataset):
    """
    Sacrifice some readability to make life easier.
    Whatever input array/argument tensor provided will be the output for dataset.
    """

    def __init__(self, *args):
        self.args = args
        for a1, a2 in zip(self.args, self.args[1:]):
            assert a1.shape[0] == a2.shape[0]

    def __getitem__(self, index):
        indexed = tuple(torch.as_tensor(a[index]) for a in self.args)
        return indexed

    def __len__(self):
        return self.args[0].shape[0]
