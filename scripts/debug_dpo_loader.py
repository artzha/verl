# scripts/debug_dpo_loader.py
import traceback
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.utils import hf_tokenizer, hf_processor
from verl.trainer.main_ppo import create_dpo_dataset, create_rl_sampler

CONFIG_DIR = "/robodata/arthurz/Research/cotnav/external/verl/verl/trainer/config"

def main():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="motion_dpo_trainer",
            overrides=[
                "algorithm.objective=dpo",
                # use your exact files here:
                "data.train_files=data/unified_motion_rm_v3_single_pair/parquet/train/train.parquet",
                "data.val_files=data/unified_motion_rm_v3_single_pair/parquet/val/val.parquet",
                # critical for readable traceback:
                "data.dataloader_num_workers=0",
                # optional for deterministic repro:
                "data.shuffle=false",
            ],
        )

    print(OmegaConf.to_yaml(cfg.data))
    model_path = 'Qwen/Qwen3-VL-2B-Instruct'
    tokenizer = hf_tokenizer(model_path, trust_remote_code=cfg.data.get("trust_remote_code", False))
    processor = hf_processor(model_path, trust_remote_code=cfg.data.get("trust_remote_code", False), use_fast=True)

    train_ds, val_ds, collate_fn = create_dpo_dataset(cfg, tokenizer, processor)
    train_sampler = create_rl_sampler(cfg.data, train_ds)

    train_loader = StatefulDataLoader(
        dataset=train_ds,
        batch_size=cfg.data.get("gen_batch_size", cfg.data.train_batch_size),
        num_workers=cfg.data.dataloader_num_workers,  # 0 for debug
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    # First: isolate bad row index directly from dataset
    for i in range(len(train_ds)):
        try:
            _ = train_ds[i]
        except Exception as e:
            print(f"\nFAILED dataset idx={i}: {type(e).__name__}: {e}")
            row = train_ds.dataframe.iloc[i].to_dict()
            for k in ["prompt", "chosen", "rejected", "pairs", "pair_valid_mask"]:
                if k in row:
                    print(f"{k}: type={type(row[k])}")
            traceback.print_exc()
            return

    # Then: verify dataloader collation path
    batch = next(iter(train_loader))
    print("Batch keys:", list(batch.keys()))
    for k, v in batch.items():
        print(k, type(v), len(v) if hasattr(v, "__len__") else "n/a")

if __name__ == "__main__":
    main()