import torch

import colossalai
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.zero.init_ctx import ZeroInitContext


from transformers import BloomForCausalLM



def BloomCoder():

    zero = hasattr(gpc.config, "zero")
    if zero:
        with ZeroInitContext(
            target_device = torch.cuda.current_device(),
            shard_strategy = gpc.config.zero.model_config.shard_strategy,
            shard_param = True
        ):
            model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3")

    