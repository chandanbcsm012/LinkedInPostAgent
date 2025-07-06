import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def read_json(filename):
    logger.info(f"Reading file {filename}")
    with open(filename, "r") as f:
        return json.load(f)