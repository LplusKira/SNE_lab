from os import environ

# For logging
SILENCE_SNE = bool(environ.get('SILENCE_SNE'))
DEBUG = bool(int(environ.get('DEBUG', 0)))  # Any int but 0 for further debugging
LogFlags = {
    'INFO': '[info] ',
    'WARN': '[warn] ',
    'USE': '[usage] ',
    'ERR': '[error] ',
}

# For test
TEST_SNE = bool(environ.get('TEST_SNE'))
TEST_DIR = 'report/.testsne/'
