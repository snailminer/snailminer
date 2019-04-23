import sys
import signal
from logging.config import dictConfig

def handler(signum, frame):
    raise SystemExit

def logging_setting():
    log_setting = {
        'version': 1,
        'formatters': {
            "verbose": {
                'format': '[%(asctime)s][%(levelname)s] %(message)s',
                'datefmt': '%m/%d/%Y %H:%M:%S',
            }
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'verbose',
            },
        },
        'root': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    }
    return log_setting

dictConfig(logging_setting())

def main():
    from snailminer.collector import run

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    return sys.exit(run())


if __name__ == '__main__':
    main()
