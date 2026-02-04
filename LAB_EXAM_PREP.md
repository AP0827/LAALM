# Lab SEE Evaluation Prep: LAALM Project (Final Comprehensive Version)

This document is maximized for **technical depth**, **innovation points**, and **comprehensive objectives** to help you score full marks and answer any curveball questions.

---

## 📝 Part 1: Write-Up (10 Marks)

**1. Problem Statement / Objectives**
*   **Primary Objective**: To architect a Noise-Robust Hybrid Audio-Visual Speech Recognition (AVSR) system capable of operating in low-SNR (Signal-to-Noise Ratio) environments.
*   **Secondary Objective**: To solve the "Cocktail Party Problem" by effectively isolating speech signals using visual lip-reading cues when acoustic signals are corrupted by overlapping speech or background noise.
*   **Technical Objective**: To implement a **Linear Late Fusion** strategy that dynamically weighs audio vs. visual confidence scores at the word level.
*   **Research Objective**: To validate the hypothesis of **"Error Orthogonality"**—demonstrating that audio and visual modalities fail in distinct, non-overlapping scenarios.
*   **Application Objective**: To deploy a Large Language Model (Llama 3) as a **Semantic Post-Processor** to correct grammatical errors and hallucinated words in the fused transcript.

**2. Dataset Description**
*   **Primary Training Data**: **LRS3-TED (Lip Reading Sentences 3)**.
    *   *Scale*: 433 hours of video.
    *   *Content*: TED Talks (high variability in pose, lighting, and speakers).
*   **Test Data**: Custom gathered real-world samples (`samples/` directory).
    *   *Conditions*: Low light, background chatter, side-angles.
*   **Data Structure**:
    *   **Video**: 25 FPS (Frames Per Second), aligned to audio.
    *   **Audio**: 16kHz Mono, 16-bit depth.
    *   **Text**: Ground truth transcripts for calculation of accuracy.

**3. Model / Algorithm Used (Technical Detail)**
*   **Architecture Type**: **Hybrid Late-Fusion Pipeline (Decision Level Fusion)**.
*   **Component Breakdown**:
    1.  **Visual Encoder (AutoAVSR)**:
        *   **Front-end**: **ResNet-18** (Extracts spatial features from lip pixels).
        *   **Back-end**: **Conformer** (Convolution + Transformer). Captures temporal dependencies (how mouth shapes change over time).
    2.  **Audio Encoder (DeepGram)**:
        *   State-of-the-art ASR utilizing End-to-End Deep Neural Networks (DNNs).
    3.  **Large Language Model (Llama 3 - 8B/70B)**:
        *   **Role**: Semantic Corrector.
        *   **Mechanism**: Zero-Shot Prompting. Input is the "noisy transcript", output is the "clean transcript".
    4.  **Fusion Algorithm**:
        *   **Confidence-Weighted Selection**: For every word time-step $t$, we select the word $w$ with maximum probability $P(w)$ from either Audio or Visual streams.

**4. Basic Workflow (Data → Model → Output)**
1.  **Input Acquisition**: Capture video file ($V$).
2.  **Modality Separation**:
    *   $V \rightarrow V_{frames}$ (Video Tensor: $T \times H \times W \times C$)
    *   $V \rightarrow A_{wave}$ (Audio Tensor: $T \times 1$)
3.  **Feature Extraction**:
    *   Visual: Mouth Region of Interest (ROI) inputs to AutoAVSR $\rightarrow$ Lip-read Text ($T_{visual}$).
    *   Audio: Waveform inputs to DeepGram $\rightarrow$ Audio Text ($T_{audio}$).
4.  **Temporal Alignment**: Align $T_{audio}$ and $T_{visual}$ based on timestamps.
5.  **Multi-Modal Fusion**: Combine streams into $T_{fused}$.
6.  **Semantic Correction**: $LLM(T_{fused}) \rightarrow T_{final}$.
7.  **Evaluate**: Compare $T_{final}$ with Ground Truth ($T_{gt}$) using WER.

**5. Innovative Components (Points for "Innovation")**
1.  **Semantic Error Correction via LLMs**: Unlike traditional AVSR which stops at the decoder, we use an LLM (Llama 3) to "reason" about the sentence. This fixes errors that are technically phonetically correct but semantically nonsensical.
2.  **N-Best Hypothesis Fusion**: We don't just fuse the top word. We capture the "Distributional Cloud" (Top-5 guesses) from the visual model and allow the LLM to pick the one that makes the most grammatical sense contextually.
3.  **Dynamic Confidence Weighting**: Our fusion algorithm isn't static (50/50). It adapts frame-by-frame. If the room is loud, it trusts eyes (Visual). If the face is turned, it trusts ears (Audio).
4.  **Error Orthogonality Exploitation**: We explicitly designed the system based on the statistical finding that Audio and Visual errors correspond to different physical phenomena (Noise vs. Occlusion), maximizing the theoretical upper bound of accuracy.

---

## 💻 Part 2: Conduction of Experiments (20 Marks)

**1. Dataset Loading and Preprocessing**
*   **Library Stack**: `torchvision` (video), `ffmpeg` (audio), `opencv` (face detection).
*   **Critical Step**: **Face Detection & Comparison**.
    *   We use **RetinaFace** to detect the face.
    *   We crop the mouth region (typically $96 \times 96$ pixels).
    *   We normalize to grayscale to reduce dimensionality.

**2. Model Implementation Steps**
*   **Step 1: Initialization**: Load model weights (`.pth` files) into GPU memory.
*   **Step 2: Inference Loop**:
    *   Process Audio $\rightarrow$ JSON response.
    *   Process Video $\rightarrow$ Beam Search Decoder $\rightarrow$ Transcript.
*   **Step 3: Fusion Function**:
    ```python
    def fuse(audio_words, visual_words):
        # Align lists by time
        # If audio_conf > visual_conf: take audio_word
        # Else: take visual_word
        return result
    ```

**3. Training and Testing**
*   **Pre-training**: Models were pre-trained on high-performance clusters (HPCs) using the LRS3 dataset.
*   **Testing Protocol**:
    *   **Metric 1: WER (Word Error Rate)**
        $$ WER = \frac{S + D + I}{N} $$
        (Substitutions + Deletions + Insertions / Total Words).
    *   **Metric 2: Latency**. Time taken from input to text.
*   **Ablation Study Results**:
    *   LLM Correction improved accuracy by **~77%**.
    *   Visual-only performance remains stable even in 0dB SNR (extreme noise).

**4. Result Generation**
*   **Demo**: Your script generates a JSON file containing the step-by-step transformation:
    *   `"audio": "The cat s_t on..."` (Low confidence on 'sat')
    *   `"visual": "... cat sat on ..."` (High confidence on 'sat')
    *   `"fused": "The cat sat on..."` (Corrected)

---

## 🗣️ Part 3: Viva-Voce (20 Marks)

### Technical Questions (The "Hard" Ones)

**Q: What is a Conformer and why use it?**
*   **A:** A Conformer combines **CNNs** (Convolutional Neural Networks) and **Transformers**. CNNs are good at extracting local shapes (curve of the lip), while Transformers are good at global sequences (sentence context). Lip reading needs both.

**Q: Why standard Transformers (like BERT) aren't enough?**
*   **A:** Standard Transformers struggle with the fine-grained local details of image frames. This is why we use ResNet (features) + Conformer (sequence).

**Q: How do you handle "Audio-Visual Synchronization"?**
*   **A:** We assume the input video file is synchronized. In the model, we use fixed framerates (25 FPS). If audio lags video, the fusion would fail. We rely on standard container formats (MP4) to maintain sync.

**Q: What is Beam Search?**
*   **A:** Instead of just picking the single best next word (Greedy Decoding), Beam Search keeps track of the 'k' most promising sentences at every step. It builds a tree of possibilities and picks the best path at the end. This is crucial for avoiding silly mistakes.

### System Design Questions

**Q: Why "Late Fusion" and not "Early Fusion"?**
*   **A:**
    *   **Early Fusion**: Concatenating Audio and Video vectors *before* processing. Hard to train, requires perfect alignment.
    *   **Late Fusion (Ours)**: Processing them separately and combining the *results*. This is more robust—if the camera breaks, the audio system still works independently. It's modular.

**Q: Why Llama - isn't it too big?**
*   **A:** We use the **8B parameter** version, which fits on consumer GPUs. For production, we would use "Distillation" to compress it or use a smaller specific model like typical Seq2Seq transformers (T5 or BART).

### Project-Specific Questions

**Q: What happens if I cover my mouth?**
*   **A:** The Visual confidence score drops to near zero. The "Confidence-Weighted Fusion" logic will automatically ignore the visual stream and rely 100% on Audio.

**Q: What happens if there is loud background noise?**
*   **A:** The Audio confidence score drops. The system will then rely more heavily on the Visual stream (Lip Reading), which is immune to acoustic noise.

**Q: What are the limitations?**
*   **A:**
    1.  **Homophones**: Words that look the same on lips (e.g., "Park" vs "Bark"). Visual-only models cannot distinguish these.
    2.  **Profile Views**: Lip reading requires a mostly frontal face. 90-degree side views fail.
    3.  **Latency**: Current pipe takes ~15-20s for a query. Real-time conversation needs <200ms.

**Q: How would you make this "Real-Time"?**
*   **A:** By using "Streaming Inference". Instead of waiting for the file to finish, we would process 1-second chunks (packets) continuously.
