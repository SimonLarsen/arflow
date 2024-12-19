from typing import Optional
from dataclasses import dataclass
import torch


@dataclass
class FlowModelOutput:
    flow_pred: torch.FloatTensor
    flow_true: torch.FloatTensor
    mask: Optional[torch.BoolTensor] = None
