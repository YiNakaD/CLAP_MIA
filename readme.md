### CLAP Model Training


#### 1. Install necessary libraries:

```bash
librosa
soundfile
accelerate
ffmpeg
torchaudio
transformers==4.45.1
```

#### 2. Data Processing(For example, using the LibriSpeech dataset):

##### 2.1.1 Download the LibriSpeech dataset from [LibriSpeech](http://www.openslr.org/12).

##### 2.1.2 Extract the dataset to the `data` folder.

##### 2.1.3 Use `data/Librispeech_process.ipynb` to preprocess audio data.

#### 3. Model Training:

The model will be saved in `checkpoints/model`.

```bash
python train.py
```

Note: Please remember to modify data path and other parameters in `train.py` before running.

### Detection

#### 1. Install necessary libraries:

```bash
cd local
pip install -r requirements.txt
```

#### 2. Generate gibberish:

```bash
python generate.py
```

#### 3. Optimize audios:

```bash
python clap_opt_1_minut.py
```

#### 4. Detect:

The sample code is in `local/vote.py`.

```bash
cd local
python vote.py
```
