from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

import torch

from typing import Union, Any, TypedDict

class Mode(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

class Quantization(Enum):
    TORCH_STATIC = 'torch-static'
    TORCH_DYNAMIC = 'torch-dynamic'
    ONNX_STATIC = 'onnx-static'
    ONNX_DYNAMIC = 'onnx-dynamic'

class DatasetOutput(TypedDict):
    data: torch.Tensor
    label: Union[str, int, torch.Tensor]

class BaseDataset(ABC):
    
    @abstractmethod
    def __init__(self, root_dir: str, mode: Mode):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    # TODO: fix return type, need to use DatasetOutput
    def __getitem__(self, idx) -> Any:
        pass

@dataclass
class TrainingArgs:
    model: str
    dataset: str
    epoch: int
    lr: float
    batchsz: int
    cuda: bool

@dataclass
class ExportingArgs:
    model: str
    checkpoint_id: str
    quant: Quantization
