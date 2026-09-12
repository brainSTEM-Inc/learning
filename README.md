# Computational Certificate

This repository contains the exact computational certificate accompanying our paper on self-directed graph prediction.

The certificate verifies the finite binary nonadjacent transfer case discussed in the manuscript. It uses only Python's standard library and exact integer arithmetic and supplements, rather than replaces, the structural proof.

## Requirements

Python 3.10 or later. No third-party packages are required.

## Run

```bash
python3 binary_nonadjacent_certificate.py
```

A successful run prints:

```text
All exact NMT certificate assertions passed.
```
