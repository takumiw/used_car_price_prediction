import getpass
import os
import socket
from datetime import datetime
from pathlib import PosixPath

from lightgbm.callback import _format_eval_result
from loguru import logger

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory


def set_logger(dir: PosixPath, level: str = "DEBUG") -> None:
    """
    Args:
        dir (PosixPath): path to log directory, e.g. .../logs/exp001/fold0/
    memo:
        logs は実験番号で管理する, exec_time はその都度発行する
        logs / experiment id / log stuff  e.g. logs/exp001/20230101_123456_fold0.log
    """
    exec_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = (dir / exec_time).with_suffix(".log")

    dir.mkdir(parents=True, exist_ok=True)
    logger.add(path, format="{time:YYYY-MM-DD THH:mm:ss} - {level} - {message}", level=level)

    # getLogger("matplotlib").setLevel(WARNING)  # Suppress matplotlib logging
    # # getLogger("requests").setLevel(WARNING)  # Suppress requests logging
    # # getLogger("urllib3").setLevel(WARNING)  # Suppress urllib3 logging
    # getLogger("cpp").setLevel(WARNING)  # Suppress cpp Warning

    logger.info(f"{getpass.getuser()}@{socket.gethostname()}:{path}")


def log_evaluation(period: int = 1, show_stdv: bool = True):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = "\t".join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.debug(f"[{env.iteration + 1}]\t{result}")

    _callback.order = 10
    return _callback
