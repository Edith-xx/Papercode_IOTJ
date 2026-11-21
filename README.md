# Toward Intelligent Lightweight and Efficient UAV Identification With RF Fingerprinting

PyTorch implementation accompanying the paper:

> **Toward Intelligent Lightweight and Efficient UAV Identification With RF Fingerprinting**  
> Zhenxin Cai, Yu Wang, Qi Jiang, Guan Gui, and Jin Sha  
> *IEEE Internet of Things Journal*, vol. 11, no. 15, pp. 26329–26339, Aug. 2024.  
> DOI: [10.1109/JIOT.2024.3395466](https://doi.org/10.1109/JIOT.2024.3395466)

---

## Overview

Radio-frequency fingerprint identification (RFFI) identifies individual wireless transmitters from device-specific hardware imperfections embedded in their emitted RF signals. For UAV monitoring, deep-learning-based RFFI can provide passive and non-cooperative device authentication, but many existing models require substantial computation and storage, which limits deployment on resource-constrained platforms.

This work proposes a **lightweight UAV RFFI framework** centered on a **Lightweight Multiscale Convolution (LMSC) block**. The LMSC block combines hierarchical multiscale feature extraction with depthwise separable convolution, enabling the network to capture informative time–frequency patterns while substantially reducing computational cost. The framework further uses **sample truncation** to reduce input length and **Mixup regularization** to improve generalization without increasing model parameters or MACs.

The task considered in the paper is **individual identification of seven identical UAVs**, rather than UAV model or flight-mode classification.

---


## Dataset

Experiments are conducted on the public **Hovering UAVs RF Fingerprinting Dataset**:

- Dataset page: [https://genesys-lab.org/hovering-uavs](https://genesys-lab.org/hovering-uavs)
- Number of UAVs: **7 identical UAV devices**
- Collection environment: **RF anechoic chamber**
- Receiver: **Ettus USRP X310** with a **UBX 160** daughterboard
- Total recorded samples reported by the dataset description: approximately **13,000**
- Approximate I/Q length reported per sample: about **92,000** samples

## Environment

The environment reported in the paper is:

```text
Python 3.6.4
PyTorch 1.10.2
```

Install the remaining packages according to the imports used by the repository or the provided `requirements.txt` file.

---

## Experimental Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 50 |
| Classification loss | Cross-entropy |
| Training SNR | Uniformly sampled from 0–10 dB |
| Test SNR | Fixed levels from −5 to 20 dB |
| Train/validation split | 8:2 |
| Full task | Seven-class UAV individual identification |

---

## Results

The average accuracy is computed over test SNRs from −5 dB to 20 dB.

| Method | MACs | Parameters | Storage (MB) | Training time (s/epoch) | Testing time (ms) | Avg. accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CVCNN | $4.84\times10^9$ | $1.25\times10^7$ | 47.78 | 85.91 | 15.31 | 90.72% |
| ResNet50 | $6.75\times10^9$ | $2.35\times10^7$ | 89.71 | 28.88 | 6.83 | 89.35% |
| MobileNet | $1.42\times10^9$ | $4.21\times10^6$ | 16.06 | 85.96 | 6.88 | 88.31% |
| ShuffleNet | $1.02\times10^9$ | $5.36\times10^6$ | 20.44 | **8.67** | 7.56 | 86.91% |
| 2-D CNN | $6.81\times10^9$ | $4.67\times10^6$ | 17.82 | 43.41 | 3.68 | 78.44% |
| Lightweight | $8.00\times10^8$ | $4.06\times10^4$ | **0.15** | 28.07 | **3.48** | 58.61% |
| Raw IQ | $1.11\times10^{10}$ | $7.26\times10^5$ | 2.77 | 36.22 | 3.83 | 26.63% |
| **Proposed** | **$1.36\times10^8$** | $1.27\times10^6$ | 4.84 | 19.47 | 3.56 | **93.25%** |
| **Proposed + Mixup** | Same | Same | Same | 19.93 | Not separately reported | **93.92%** |

The proposed model has the lowest MAC count among the compared methods. The paper reports that even the next-lowest-complexity baseline requires approximately **5.88×** more MACs.

---

## Citation

Please cite the following article if this repository is useful in your research:

```bibtex
@article{cai2024toward,
  author  = {Cai, Zhenxin and Wang, Yu and Jiang, Qi and Gui, Guan and Sha, Jin},
  title   = {Toward Intelligent Lightweight and Efficient UAV Identification With RF Fingerprinting},
  journal = {IEEE Internet of Things Journal},
  volume  = {11},
  number  = {15},
  pages   = {26329--26339},
  year    = {2024},
  doi     = {10.1109/JIOT.2024.3395466}
}
```

---

## Acknowledgements

This project uses the public Hovering UAVs RF Fingerprinting Dataset. We thank the dataset providers and the developers of PyTorch and the open-source scientific Python ecosystem.

The LMSC design is inspired by multiscale representation learning in Res2Net and adopts depthwise separable convolution to reduce computational cost.

---

## Contact

For questions about the paper, please contact the authors:

- Zhenxin Cai: `652022230002@smail.nju.edu.cn`

---

## License

This repository is intended for academic research. Please refer to the repository's `LICENSE` file for the code license. The dataset is subject to the terms specified by its original provider.
