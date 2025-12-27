# LAALM - Lip-reading and Audio Analysis with LLM# LipNet: End-to-End Sentence-level Lipreading



Multi-modal speech transcription system combining audio analysis, visual lip-reading, and LLM correction.Keras implementation of the method described in the paper 'LipNet: End-to-End Sentence-level Lipreading' by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas (https://arxiv.org/abs/1611.01599).



## Features![LipNet performing prediction (subtitle alignment only for visualization)](assets/lipreading.gif)



- **Audio Transcription**: DeepGram API for high-accuracy speech-to-text## Results

- **Visual Lip-reading**: LipNet neural network for video-based transcription

- **LLM Fusion**: Groq/OpenAI for intelligent multi-modal transcript correction|        Scenario        | Epoch |  CER  |  WER  |  BLEU  |

- **Confidence Scoring**: Word-level confidence for both audio and visual inputs| :---------------------: | :---: | :---: | :----: | :----: |

- **Automatic Fallback**: Mock mode when APIs unavailable|   Unseen speakers [C]   |  N/A  |  N/A  |  N/A  |  N/A  |

|     Unseen speakers     |  178  | 6.19% | 14.19% | 88.21% |

## Quick Start| Overlapped speakers [C] |  N/A  |  N/A  |  N/A  |  N/A  |

|   Overlapped speakers   |  368  | 1.56% | 3.38% | 96.93% |

```bash

# Setup (see SETUP.md for detailed instructions)**Notes**:

source .venv/bin/activate

pip install -r requirements.txt- [C] means using curriculum learning.

- N/A means either the training is in progress or haven't been performed.

# Configure API keys- Your contribution in sharing the results of this model is highly appreciated :)

cp .env.example .env

# Edit .env with your GROQ_API_KEY, DEEPGRAM_API_KEY, etc.## Dependencies



# Run* Keras 2.0+

python main.py* Tensorflow 1.0+

```* PIP (for package installation)



## Usage ExamplePlus several other libraries listed on `setup.py`



```python## Usage

from pipeline import run_mvp

To use the model, first you need to clone the repository:

result = run_mvp(

    video_file="samples/video/bbaf2n.mpg",```

    audio_file="samples/audio/swwp4p.wav",git clone https://github.com/rizkiarm/LipNet

    lipnet_weights="LipNet/evaluation/models/unseen-weights178.h5"```

)

Then you can install the package:

print(f"Final Transcript: {result['final_transcript']}")

print(f"DeepGram: {result['deepgram']['transcript']}")```

print(f"LipNet: {result['lipnet']['transcript']}")cd LipNet/

```pip install -e .

```

## Project Structure

**Note:** if you don't want to use CUDA, you need to edit the ``setup.py`` and change ``tensorflow-gpu`` to ``tensorflow``

```

LAALM/You're done!

├── main.py                 # Entry point

├── pipeline.py             # MVP orchestrationHere is some ideas on what you can do next:

├── test.py                 # Testing script

├── load_env.py             # Environment configuration* Modify the package and make some improvements to it.

├── requirements.txt        # Python dependencies* Train the model using predefined training scenarios.

├── .env.example            # API key template* Make your own training scenarios.

├── DeepGram/               # Audio transcription module* Use [pre-trained weights](https://github.com/rizkiarm/LipNet/tree/master/evaluation/models) to do lipreading.

│   ├── transcriber.py* Go crazy and experiment on other dataset! by changing some hyperparameters or modify the model.

│   └── word_confidence.py

├── LipNet/                 # Visual lip-reading module## Dataset

│   ├── evaluation/

│   │   ├── models/         # Pre-trained weightsThis model uses GRID corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)

│   │   └── predict_with_confidence.py

│   └── lipnet/## Pre-trained weights

│       └── model.py

├── Transformer/            # LLM correction moduleFor those of you who are having difficulties in training the model (or just want to see the end results), you can download and use the weights provided here: https://github.com/rizkiarm/LipNet/tree/master/evaluation/models.

│   └── llm_corrector.py

└── samples/                # Test dataMore detail on saving and loading weights can be found in [Keras FAQ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

    ├── audio/*.wav         # Audio samples

    └── video/*.mpg         # Video samples## Training

```

There are five different training scenarios that are (going to be) available:

## Models

### Prerequisites

Pre-trained LipNet models (in `LipNet/evaluation/models/`):

- **unseen-weights178.h5** - For unseen speakers (14.19% WER)1. Download all video (normal) and align from the GRID Corpus website.

- **overlapped-weights368.h5** - For overlapped speakers (3.38% WER)2. Extracts all the videos and aligns.

3. Create ``datasets`` folder on each training scenario folder.

## Requirements4. Create ``align`` folder inside the ``datasets`` folder.

5. All current ``train.py`` expect the videos to be in the form of 100x50px mouthcrop image frames.

- Python 3.11+ (required for TensorFlow 2.12/Keras 2.12 compatibility)   You can change this by adding ``vtype = "face"`` and ``face_predictor_path`` (which can be found in ``evaluation/models``) in the instantiation of ``Generator`` inside the ``train.py``

- CUDA-compatible GPU (optional, for faster inference)6. The other way would be to extract the mouthcrop image using ``scripts/extract_mouth_batch.py`` (usage can be found inside the script).

- API Keys: DeepGram, Groq (or OpenAI)7. Create symlink from each ``training/*/datasets/align`` to your align folder.

8. You can change the training parameters by modifying ``train.py`` inside its respective scenarios.

## API Keys

### Random split (Unmaintained)

Set in `.env` file:

```Create symlink from ``training/random_split/datasets/video`` to your video dataset folder (which contains ``s*`` directory).

GROQ_API_KEY=your_groq_key

DEEPGRAM_API_KEY=your_deepgram_keyTrain the model using the following command:

OPENAI_API_KEY=your_openai_key  # Optional fallback

``````

./train random_split [GPUs (optional)]

## License```



See individual module directories for specific licenses.**Note:** You can change the validation split value by modifying the ``val_split`` argument inside the ``train.py``.


### Unseen speakers

Create the following folder:

* ``training/unseen_speakers/datasets/train``
* ``training/unseen_speakers/datasets/val``

Then, create symlink from ``training/unseen_speakers/datasets/[train|val]/s*`` to your selection of ``s*`` inside of the video dataset folder.

The paper used ``s1``, ``s2``, ``s20``, and ``s22`` for evaluation and the remainder for training.

Train the model using the following command:

```
./train unseen_speakers [GPUs (optional)]
```

### Unseen speakers with curriculum learning

The same way you do unseen speakers.

**Note:** You can change the curriculum by modifying the ``curriculum_rules`` method inside the ``train.py``

```
./train unseen_speakers_curriculum [GPUs (optional)]
```

### Overlapped Speakers

Run the preparation script:

```
python prepare.py [Path to video dataset] [Path to align dataset] [Number of samples]
```

**Notes:**

- ``[Path to video dataset]`` should be a folder with structure: ``/s{i}/[video]``
- ``[Path to align dataset]`` should be a folder with structure: ``/[align].align``
- ``[Number of samples]`` should be less than or equal to ``min(len(ls '/s{i}/*'))``

Then run training for each speaker:

```
python training/overlapped_speakers/train.py s{i}
```

### Overlapped Speakers with curriculum learning

1. Copy the ``prepare.py`` from ``overlapped_speakers`` folder to ``overlapped_speakers_curriculum`` folder,
   and run it as previously described in overlapped speakers training explanation.

Then run training for each speaker:

```
python training/overlapped_speakers_curriculum/train.py s{i}
```

**Note:** As always, you can change the curriculum by modifying the ``curriculum_rules`` method inside the ``train.py``

## Evaluation

To evaluate and visualize the trained model on a single video / image frames, you can execute the following command:

```
./predict [path to weight] [path to video]
```

**Example:**

```
./predict evaluation/models/overlapped-weights368.h5 evaluation/samples/id2_vcd_swwp2s.mpg
```

## Work in Progress

This is a work in progress. Errors are to be expected.
If you found some errors in terms of implementation please report them by submitting issue(s) or making PR(s). Thanks!

**Some todos:**

- [X] Use ~~Stanford-CTC~~ Tensorflow CTC beam search
- [X] Auto spelling correction
- [X] Overlapped speakers (and its curriculum) training
- [ ] Integrate language model for beam search
- [ ] RGB normalization over the dataset.
- [X] Validate CTC implementation in training.
- [ ] Proper documentation
- [ ] Unit tests
- [X] (Maybe) better curriculum learning.
- [ ] (Maybe) some proper scripts to do dataset stuff.

## License

MIT License
