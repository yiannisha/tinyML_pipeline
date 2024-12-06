import os

import torch

from _types import ExportingArgs, TrainingArgs

from typing import Optional, Union, Tuple

rootdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def validate_training_args(args: TrainingArgs) -> None:

    if not args.model or not args.dataset:
        raise ValueError('model and dataset are required')

    if not os.path.exists(os.path.join(rootdir, 'models', f'{args.model}.py')):
        raise ValueError(f'No such model in /models: {args.model}.py')

    if not os.path.exists(os.path.join(rootdir, 'datasets', f'{args.dataset}.py')):
        raise ValueError(f'No such dataset in /datasets: {args.dataset}.py')

def validate_exporting_args(args: ExportingArgs) -> None:

    if not args.model or not args.quant:
        raise ValueError('model and quantization type are required')

    if not os.path.exists(os.path.join(rootdir, 'checkpoints', args.model)):
        raise ValueError(f'No such model in /checkpoints: {args.model}')

def get_checkpoint(model: str, id: str, ext: str) -> Optional[str]:
    checkpoints = os.listdir(os.path.join(rootdir, 'checkpoints', model))
    for i in checkpoints:
        if i.endswith(f'{id}.{ext}'):
           return os.path.join(rootdir, 'checkpoints', model, i)
    return None


def pytorch2onnx(model: Union[str, torch.nn.Module], out_path: str, shape: Tuple[int, int], verbose: bool = False) -> None:

    if isinstance(model, str):
        model: torch.nn.Module = torch.load(model, map_location=torch.device('cpu'))

    dummy_input = torch.randn(1, 1, shape[0], shape[1])

    torch.onnx.export(model, dummy_input, out_path, verbose=verbose)
