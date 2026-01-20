
# HeartMuLa – Lazy Load Patch for ComfyUI (12GB RTX 3060)

**Local music generation on NVIDIA GPUs with only 12GB VRAM.**

This repository provides a small but effective **Lazy Load patch for HeartMuLa**, enabling the model to run inside **ComfyUI** on **low-VRAM consumer GPUs**, such as the NVIDIA RTX 3060 (12GB).

The patch significantly improves inference performance and makes it possible to generate **music tracks longer than 4 minutes** on entry- and mid-range graphics cards.

---

## 🚀 Motivation

By default, HeartMuLa often falls back to running large parts of the model on the CPU when VRAM is limited.
This results in extremely long inference times:

> **Up to 2 hours for a 2-minute song** was common.

This patch avoids full CPU execution by **sequentially loading only the required SafeTensor networks**, dramatically reducing memory usage while maintaining GPU acceleration.

---

## 🧠 How It Works

The optimization is achieved through **lazy, sequential loading** of model components:

1. **Text Tokenization**
   The input text is processed with minimal memory overhead.

2. **Audio Stream Decoding**
   Model weights are loaded only when required and released afterward.

This approach allows efficient inference without loading the entire model into VRAM at once.

---

## 🛠 Installation

> **Recommended:** Create a backup of your existing HeartMuLa ComfyUI plugin before proceeding.

1. Locate your current **HeartMuLa ComfyUI plugin**.
2. Replace the **two corresponding files** with the versions provided in this repository.
3. Restart ComfyUI.

No additional configuration is required.

---

## 📁 Directory Structure

```text
ComfyUI/
└── custom_nodes/
    └── HeartMuLa_ComfyUI/
        ├── __init__.py
        └── util/
            └── heartlib/
                └── pipelines/
                    └── music_generation.py
```

---

## 🔗 Sources & References

* **HeartMuLa Core Library**
  [https://github.com/HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib)

* **HeartMuLa ComfyUI Plugin**
  [https://github.com/benjiyaya/HeartMuLa_ComfyUI](https://github.com/benjiyaya/HeartMuLa_ComfyUI)

* **Original Lazy Load Patch (Pull Request)**
  [https://github.com/HeartMuLa/heartlib/pull/5](https://github.com/HeartMuLa/heartlib/pull/5)

---

## 🔮 Future Outlook

It may be possible to **integrate the `Lazy_Load` functionality directly into the ComfyUI GUI**, making it selectable via the interface and easier to use for non-technical users.

Contributions and ideas are welcome.

---

## 📜 License

Please refer to the original HeartMuLa repositories for licensing details.

---

**Enjoy faster local music generation on low-VRAM GPUs! 🎶**


