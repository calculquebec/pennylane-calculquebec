import logging

logger = logging.getLogger("pennylane_calculquebec")
logger.setLevel(logging.INFO)

handler = logging.FileHandler("pennylane_calculquebec.log", mode="a", encoding="utf-8")
formatter = logging.Formatter(
    'The plugin version is 0.6.1 | %(asctime)s | Incident: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

