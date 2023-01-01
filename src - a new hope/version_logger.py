from pathlib import Path
from time import localtime, strftime
import torch.nn as nn
import torch
import io


def get_local_time() -> str:
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


class VersionLogger:
    def __init__(self, path: Path, print_to_console=False):
        self.path = path
        self.print_to_console = print_to_console
        self.log_path = Path(path, './logs.log')
        self.models_path = Path(path, './models/')

        self.path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

    def print(self, *args, **kwargs):
        output = io.StringIO()
        print(*args, file=output, **kwargs)
        contents = output.getvalue()
        output.close()

        if self.print_to_console:
            print(contents)
        with open(self.log_path, 'a') as file:
            file.write(f'{get_local_time()} - ' + contents + '\n')

    def save_model(self, model: nn.Module, name: str):
        model_path = Path(self.models_path, name)
        torch.save(model.state_dict(), model_path)
