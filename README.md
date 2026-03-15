# Kalmanorix 🔀

**Dynamic fusion of specialised embedding models using Kalman filtering**

Kalmanorix is a framework for composing multiple domain-specialised embedding models into a single, adaptive representation. Instead of training ever-larger monolithic models, it treats each specialist as providing a 'measurement' of the true semantic state, complete with an uncertainty estimate. A Kalman filter optimally fuses these measurements in real time, weighted by each model's confidence for the current query.

This is the reference implementation of the **Kalman Embedding Fusion Framework (KEFF)** concept.

## 🚀 Quick Start

```bash
pip install kalmanorix
