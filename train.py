from typing import cast, Optional, Sequence
from pathlib import Path
import argparse
import torch
from torchvision.transforms.functional import to_pil_image
from frogbox import read_json_config, SupervisedPipeline, SupervisedConfig
import accelerate
import wandb


def parse_arguments(
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, default="configs/example.json"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-keys", type=str, nargs="+")
    parser.add_argument(
        "--logging",
        type=str,
        choices=["online", "offline"],
        default="online",
    )
    parser.add_argument("--wandb-id", type=str, required=False)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--group", type=str)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments()
    config = cast(SupervisedConfig, read_json_config(args.config))

    accelerate.utils.set_seed(1234)
    pipeline = SupervisedPipeline(
        config=config,
        checkpoint=args.checkpoint,
        checkpoint_keys=args.checkpoint_keys,
        logging=args.logging,
        wandb_id=args.wandb_id,
        tags=args.tags,
        group=args.group,
        evaluator_output_transform=lambda x, y, y_pred: (
            y_pred.flow_pred.contiguous(),
            y_pred.flow_true,
        ),
    )

    def log_images(pipeline: SupervisedPipeline):
        wandb_images = []
        generator = torch.Generator(pipeline.device).manual_seed(1234)
        outputs = pipeline.model.sample(
            num_images=8,
            num_flow_iterations=30,
            classes=[0, 1, 2, 3, 0, 1, 2, 3],
            progress=False,
            generator=generator,
        )

        for image in outputs:
            wandb_images.append(wandb.Image(to_pil_image(image)))
        pipeline.log({"test/images": wandb_images})

    pipeline.install_callback(
        event=pipeline.log_interval,
        callback=log_images,
        only_main_process=True,
    )

    pipeline.run()
