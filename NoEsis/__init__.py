"""Init file for NoEsis package."""  # pylint: disable=invalid-name
import datetime
import logging
import transformers  # pylint: disable=unused-import

logger = logging.getLogger('transformers')
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(
    f"log/{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_test.log"))
logger.info("Logger set up")
