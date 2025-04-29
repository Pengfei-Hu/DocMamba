# DocMamba: Efficient Document Pre-training with State Space Model

This is the official implementation of our AAAI 2025 paper:  
[**DocMamba: Efficient Document Pre-training with State Space Model**](https://ojs.aaai.org/index.php/AAAI/article/view/34584)

<p align="center">
  <img src="./imgs/logo.png" width="120"/>
</p>

---

## 🧾 Overview

**DocMamba** is a state space model (SSM)-based framework designed for Visually-rich Document Understanding (VrDU). By replacing the quadratic-complexity Transformer with a linear-time **Mamba** encoder, DocMamba offers superior efficiency and robust generalization to longer sequences.

### 🔍 Key Features

- 🚀 **Linear-time computation** with respect to input length
- 🧠 **Effective long-context modeling** and **length extrapolation**
- 📊 **Competitive performance** against strong Transformer baselines (LayoutLMv3) on FUNSD, CORD, and SROIE

---

### 📦 Dependencies

Core requirements include:

- [**Mamba**](https://github.com/state-spaces/mamba)
- [**Transformers**](https://github.com/huggingface/transformers)

---

## 🛠️ Usage


### Step 1: Data Processing

Prepare the training data using:

```bash
python data_process/generate_data.py
```

Modify it as needed to adapt to your own datasets.

### Step 2: Pretraining

Launch pretraining with:

```bash
bash scripts/pretrain.sh
```

Again, feel free to adjust parameters in the script to fit your data and training setup.

### 🔹 Quick Example

For a minimal example of the model architecture with reduced dependencies, refer to:

```
./runner/example.py
```


---

## 🧪 Tips & Tricks

The following strategies are crucial to ensure stable and efficient training:

- ✅ **Whole Word Masking (WWM) Refinement**: Following standard MLM pre-training, an additional round using WWM further boosts model performance. See `MLMProcessor` in `./libs/data/gma/transform.py`
- ✅ **Avoid `<pad>` Tokens** during batching: Do **not** pad short sequences up to the longest in a batch. Instead, **truncate longer sequences** to match the shortest. This avoids instability and NaN issues during training. See `TokenProcessor` in `./libs/data/gma/transform.py`
- ✅ **Dynamic Bucketing Strategy**: Input sequences are grouped into non-overlapping buckets based on their lengths, with each bucket covering a fixed range of 64. Within each bucket, sequences are uniformly truncated to the same length, and the batch size is dynamically adjusted according to the formula `b = k / l`, where `k` is a tunable constant, `l` is the input length. See `BucketSampler` in  `./libs/data/gma/bucket_sampler.py`

---

## 📚 Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{hu2025docmamba,
  title={Docmamba: Efficient document pre-training with state space model},
  author={Hu, Pengfei and Zhang, Zhenrong and Ma, Jiefeng and Liu, Shuhang and Du, Jun and Zhang, Jianshu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={22},
  pages={24095--24103},
  year={2025}
}
```
