import os

# for parsing data, init params
USR_LABELS_FIELDS = map(lambda v: int(v), os.environ.get('USR_LABELS_FIELDS').split(',')) if('USR_LABELS_FIELDS' in os.environ) else [1] # default, only use the first one-hot encoded labels
USR_LABELS_FIELDS.sort()

USR_TOTAL_LABELS_FIELDS = int(os.environ.get('USR_TOTAL_LABELS_FIELDS')) if('USR_TOTAL_LABELS_FIELDS' in os.environ) else 5
AGES_BOUNDERIES = [17, 35, 65, 100]

# for training
MAX_TRAIN_NUM = int(os.environ.get('MAX_TRAIN_NUM')) if('MAX_TRAIN_NUM' in os.environ) else 10
LEARNING_RATE = float(os.environ.get('LEARNING_RATE')) if('LEARNING_RATE' in os.environ) else 0.1
LAMBDA = float(os.environ.get('LAMBDA')) if('LAMBDA' in os.environ) else 0.1

# for debugging
DEBUG = int(os.environ.get('DEBUG')) if('DEBUG' in os.environ) else 0
