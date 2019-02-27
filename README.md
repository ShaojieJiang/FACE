# Intro

Frequency-Aware Cross-Entropy (FACE) is a simple yet effective algorithm that helps to improve the response diversity of Seq2Seq-based chatbots. The main idea is to assign token frequency-based weights to cross-entropy loss function, so as to suppress meaningless high-frequency tokens, which we believe to have caused generic responses like _"I don't know"_. Read our [paper](https://arxiv.org/abs/1902.09191) for more details.

This repo contains the official implementation of the models proposed
in paper **Improving Neural Response Diversity with Frequency-Aware
Cross-Entropy Loss**, together with the data we used for experiments.

## Requirements

To use this programme, you need [PyTorch](https://pytorch.org/) 1.0.0
(tested with Python 3.6+) and the latest version of
[ParlAI](https://github.com/facebookresearch/ParlAI) framework (tested
on commit `c6745203`).

After downloading and installing ParlAI, move the directory `face` in
this repo to `Your_Path_To_ParlAI/parlai/agents/`.  Read on for some
examples of running experiments.

## Arguments

To use different versions of `FACE`, try the following arguments:

- **`-wt` or `--weighing-time`**: Supported values are `pre`, `post`, and `none`, which correspond to "pre weight" (Eq. 10 in paper), "post weight" (Eq. 11) and "no frequency-based weights" (in cases of using CP* methods only), respectively.
- **`-ft` or `--frequency-type`**: Supported values are `out`, `gt`, and `none`, corresponding to "output frequency", "GT frequency" and "no frequency" (in cases of using CP* methods only).
- **`-cp` or `--confidence-penalty`**: Supports values like `cp` (Eq. 12), `cpf` (Eq. 13), `cpfw` (Eq. 14), and `none` if not intending to use confidence-penalty. *N.B.*, `cpfwn` is a new version of `cpfw` that normalizes weight values to the range of `[1, +inf]`, while `cpfw` weights can never approach `1`.
- **`-b` or `--beta`**: Penalty strength specifically used by `cp` (Eq. 12).
- **Seq2Seq**: if all the values of `-wt`, `-ft`, `-cp` are set to `none` (and regardless of the value of `-b`), the programme reverts to its simplest form: _Seq2Seq_.

## Examples

After installing ParlAI and FACE, change your working directory to ParlAI root.

The following example runs `FACE` with **"pre weight"** and **"output frequency"**, as is reported our best-performming model (on `OpenSubtitles` dataset, with `batch_size = 32`, and a validation period of `30s` of training):
```
python examples/train_model.py -t OpenSubtitles -m face -mf /tmp/model_face -bs 32 -vtim 30
```

To get a complete cheatsheet of `ParlAI`- or `FACE`-specific arguments, run the following command:
```
python examples/train_model.py -m face -h
```

## Data

In cases when you want to reproduce the results reported in our paper,
decompress `data.zip`. Then the subdirectories:

- **OSDb**: Contains validation and test sets used in our experiments for the OpenSubtitles dataset. We applogize that the training set is too large to share online.
- **Tweet_IDs**: Contains the Tweet IDs for `Train`, `Valid`, `Test` and `Human_eval` sets used in our experiments.

## Citation

If you use the materials provided in this repo, please cite our
following paper:

```
@inproceedings{jiang2019improving,
	Author = {Jiang, Shaojie and Ren, Pengjie and Monz, Christof and de Rijke, Maarten},
	Booktitle = {The Web Conference},
	Title = {Improving Neural Response Diversity with Frequency-Aware Cross-Entropy Loss},
	Year = {2019}}
```

## Baseline Models

I'm cleaning up the baseline models used in our paper, namely `MMI`
and `MHAM` models and their variants. There will be separate repos for
these models, but before that, you can find my current implementation
[here](https://github.com/ShaojieJiang/FACE_orig) if you really want them :wink:
