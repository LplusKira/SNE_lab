import os

USR_LABELS_FIELDS = map(lambda v: int(v), os.environ.get('USR_LABELS_FIELDS').split(',')) if('USR_LABELS_FIELDS' in os.environ) else [1] # default, only use the first one-hot encoded labels
USR_LABELS_FIELDS.sort()
