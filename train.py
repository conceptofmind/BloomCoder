import torch

import colossalai
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.core import global_context as gpc
from colossalai.zero.init_ctx import ZeroInitContext

from utils.build_dataloader import build_dataloaders
from utils.hugging_face_config import CFG

from transformers import BloomForCausalLM



def BloomCoder(cfg: CFG):

    colossalai.launch_from_torch(config='./utils/colossalai_config.py')

    assert hasattr(gpc.config, "EPOCHS"), "Please provide EPOCHS in your configuration"
    assert hasattr(gpc.config, "LEARNING_RATE"), "Please provide LEARNING_RATE in your configuration"
    assert hasattr(gpc.config, "gradient_accumulation"), "Please provide gradient_accumulation in your configuration"
    assert hasattr(gpc.config, "clip_grad_norm"), "Please provide clip_grad_norm in your configuration"

    if hasattr(gpc.config, "zero"):
        with ZeroInitContext(
            target_device = torch.cuda.current_device(),
            shard_strategy = gpc.config.zero.model_config.shard_strategy,
            shard_param = True
        ):
            model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3")
    else:
        model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3")

    # build dataloaders
    train_dataloader, eval_dataloader = build_dataloaders(cfg)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=gpc.config.LEARNING_RATE
    )

    #initialize the model
    
    engine, train_dataloader, eval_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        None,
        train_dataloader,
        eval_dataloader
    )

    steps = 0

    # training loop
    for _ in range(gpc.config.EPOCHS):
        engine.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            engine.zero_grad()
            output = model(**batch)
            
            loss = output.loss
            
            engine.backward(loss)
            engine.step()

            steps += 1

            # validation loop    
            # engine.eval()
            # for step, batch in enumerate(eval_dataloader):
            #     with torch.no_grad():
            #         batch = {k: v.cuda() for k, v in batch.items()}
            #         output = model(**batch)
            #     eval_loss = output.loss
                


BloomCoder(CFG())