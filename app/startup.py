import src.utils.randomness_utils as RandomnessUtils
from src.config.configuration_manager import ConfigurationManager


def configure():
    # Load settings

    SETTINGS = ConfigurationManager.load()

    # Seed everything
    RandomnessUtils.seed_everything(SETTINGS.random_state)
    return SETTINGS
