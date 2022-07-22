from colossalai.amp import AMP_TYPE
from colossalai.zero.shard_utils import TensorShardStrategy

# Colossal AI Global Config

EPOCHS = 1
LEARNING_RATE = 0.001

zero = dict(
    model_config = dict(
        shard_strategy = TensorShardStrategy(),
        tensor_placement_policy = 'auto',
        reuse_fp16_shard = True
    ),
)

gradient_accumulation = 1.0
clip_grad_norm = 1.0