from os import environ
import logging

# For logging
DEBUG2LOG_LEVEL = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}
DEBUG = environ.get('DEBUG')  # Logging level
LOG_LEVEL = DEBUG2LOG_LEVEL.get(DEBUG, DEBUG2LOG_LEVEL['INFO'])

# For test (indep from logging level)
TEST_SNE = bool(environ.get('TEST_SNE'))
TEST_DIR = 'report/.testsne/'
