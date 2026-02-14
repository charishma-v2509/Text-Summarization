# training/config.py

# Dataset
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
TRAIN_SAMPLES = 10000      # keep small for training from scratch
VAL_SAMPLES = 1000

# Tokenization
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

VOCAB_SIZE = 16000      # max vocabulary size

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
         

# Sequence lengths
MAX_INPUT_LENGTH = 512     # article length
MAX_OUTPUT_LENGTH = 128    # summary length

# Training (we'll use later)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
