# General-Purpose LLM Fine-Tuning Demo Codebase

This codebase contains 3 training scripts for MNIST digit classification.

## Prerequisite

**Install MNIST dataset**, decompress the files into the -ubyte files. The training data should be about 47.0 MB and the test data should be about 7.4 MB.

Run the following command to confirm correct visualizations
```
python visualize_mnist.py --mnist_dir [PATH_TO_MNIST]
```

## Fine-tune Llama-7B

```
python finetune_llama_7b.py --mnist_dir [PATH_TO_MNIST]
```

## Train Llama-7B from scratch

```
python scratch_llama_7b.py --mnist_dir [PATH_TO_MNIST]
```

## Train MLP on MNIST

```
python train_mlp.py --mnist_dir [PATH_TO_MNIST]
```