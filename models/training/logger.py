import os
import logging


def create_logger(path, file):
    """Create a log file to record the experiment's logs,
    also create and return logger

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    file_handler = logging.FileHandler(log_file)

    # set the logging level for log file
    file_handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    file_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    return logger
