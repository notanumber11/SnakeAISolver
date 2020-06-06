import logging

import absl.logging


def get_module_logger(mod_name):
    """
    To use this, do logger = get_module_logger(__name__)
    """
    ####
    # Trick to avoid: WARNING: Logging before flag parsing goes to stderr.
    # https://github.com/tensorflow/tensorflow/issues/26691
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    # End of trick
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(name)-2s] %(levelname)-4s %(message)s')
    handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
