import os

# modify by data format
USR_TOTAL_LABELS_FIELDS = int(os.environ.get('USR_TOTAL_LABELS_FIELDS')) if('USR_TOTAL_LABELS_FIELDS' in os.environ) else (4 + 21 + 2) # age/occupation/gender
AGES_BOUNDERIES = [17, 35, 65, 100]

# for training
ITEM_FIELDS_NUM = int(os.environ.get('ITEM_FIELDS_NUM')) if('ITEM_FIELDS_NUM' in os.environ) else 4
MAX_TRAIN_NUM = int(os.environ.get('MAX_TRAIN_NUM')) if('MAX_TRAIN_NUM' in os.environ) else 100
LEARNING_RATE = float(os.environ.get('LEARNING_RATE')) if('LEARNING_RATE' in os.environ) else 0.0001 # update = (1 *             OR          / MOMENTUM) * LEARNING_RATE * gradient
MOMENTUM = float(os.environ.get('MOMENTUM')) if('MOMENTUM' in os.environ) else 2.0                   #             ^ if last time loss drops ^ if last time loss increases
LAMBDA = float(os.environ.get('LAMBDA')) if('LAMBDA' in os.environ) else 0.001

# for debugging
DEBUG = int(os.environ.get('DEBUG')) if('DEBUG' in os.environ) else 0
