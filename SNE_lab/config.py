from os import environ

# For logging
DEBUG = bool(int(environ.get('DEBUG', 0)))  # Any int but 0 for further debugging
LogFlags = {
    'INFO': '[info] ',
    'WARN': '[warn] ',
    'USE': '[usage] ',
    'ERR': '[error] ',
}
