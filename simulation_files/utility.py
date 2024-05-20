import importer
from importer import import_all_modules
directory_path = 'simulation_files'
import_all_modules(directory_path, recursive=True)

class MemoryHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))

class JsonFormatter(logging.Formatter):
    def format(self, record):
        message = record.msg
        if isinstance(message, dict):
            return json.dumps(message)
        return json.dumps({"message": message})