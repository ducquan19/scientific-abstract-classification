from src.config.appsettings import AppSettings


class ConfigurationManager:
    @staticmethod
    def load() -> AppSettings:
        return AppSettings()
