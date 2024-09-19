from typing import ChainMap
import hydra
import os
import os.path as osp
import multiprocessing
import ray
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import math
import random
import torch
from omegaconf import DictConfig, OmegaConf
from src.utils.ray_utils import ProgressBar, chunks, chunks_balance
from src.utils.metric_utils import aggregate_multi_scene_metrics
from src import DetectorFreeSfM


def cfg_constrain(args):
    if args.neuralsfm.redo_matching:
        args.neuralsfm.redo_sfm, args.neuralsfm.redo_refine = True, True
    elif args.neuralsfm.redo_sfm:
        args.neuralsfm.redo_refine = True

    if args.use_prior_intrin:
        if args.colmap_cfg.no_refine_intrinsics is not True:
            logger.warning(
                f"Prior pose is provided, however no_refine_intrinsics is False and COLMAP will also optimize intrinics"
            )
    assert not (args.ray.enable and args.sub_use_ray), "Currently only support either global ray or local ray. Use both simutaneously will lead to errorness GPU assignment."
    return args


def eval_core(scene_paths, cfg, worker_id=0, pba=None):
    logger.info(
        f"Worker: {worker_id} will process: {scene_paths}, total: {len(scene_paths)} scenes"
    )

    # Make sub ray configs:
    if cfg.sub_use_ray:
        n_gpus = (
            torch.cuda.device_count()
            if not cfg.ray.enable
            else cfg.ray.n_gpus_per_worker
        )

        n_cpus = multiprocessing.cpu_count() if not cfg.ray.enable else cfg.ray.n_cpus_per_worker
        ray_cfg = {
            "slurm": False,
            "n_workers": cfg.sub_ray_n_worker,
            "n_cpus_per_worker": max(1, n_cpus / cfg.sub_ray_n_worker),
            "n_gpus_per_worker": n_gpus / cfg.sub_ray_n_worker,
            "local_mode": False,
        }
        logger.info(f"Sub ray use {max(1, n_cpus / cfg.sub_ray_n_worker)} CPUs, {n_gpus / cfg.sub_ray_n_worker} GPUs")
    else:
        ray_cfg = None

    results = {}
    # Edit(Fitz): add sfm_result_paths
    sfm_result_paths = set()

    scene_paths = tqdm(scene_paths) if pba is None else scene_paths
    for scene_path in scene_paths:
        if 'phase' not in cfg or cfg.phase == 'reconstruction':
            # Edit(Fitz): add sfm_result_paths
            # metric_dict = DetectorFreeSfM(
            dataset_name, scene_name = scene_path.split('/')[-2], scene_path.split('/')[-1]
            scene_output_dir = Path(cfg.results_output_dir) / dataset_name / scene_name
            metric_dict, sfm_result_path = DetectorFreeSfM(
                cfg.neuralsfm,
                method=cfg.method,
                work_dir=scene_path,
                # Edit(Fitz): add sfm_result_paths
                
                results_output_dir=scene_output_dir,
                gt_pose_dir=osp.join(scene_path, "poses"),
                prior_intrin_dir=osp.join(scene_path, "intrins")
                if cfg.use_prior_intrin or cfg.neuralsfm.triangulation_mode
                else None,
                prior_pose_dir=osp.join(scene_path, "poses") if cfg.neuralsfm.triangulation_mode else None,
                colmap_configs=cfg.colmap_cfg,
                use_ray=cfg.sub_use_ray,
                ray_cfg=ray_cfg,
                visualize=cfg.visualize,
                verbose=cfg.verbose,
            )
            # Edit(Fitz): add sfm_result_paths
            sfm_result_paths.add(sfm_result_path)
        else:
            raise NotImplementedError
        results[osp.basename(scene_path)] = metric_dict
        logger.info(f"Finish Processing {scene_path}.")
        if pba is not None:
            pba.update.remote(1)
    logger.info(f"Worker {worker_id} finish!")
    # Edit(Fitz): add sfm_result_paths
    # return results
    return results, sfm_result_paths

@ray.remote(num_cpus=1, num_gpus=1, max_calls=1)  # release gpu after finishing
def eval_core_ray_wrapper(*args, **kwargs):
    try:
        return eval_core(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error captured:{e}")

def eval_dataset(cfg):
    OmegaConf.resolve(cfg)
    # NOTE: Important for use relative path in hydra
    os.chdir(hydra.utils.get_original_cwd())
    cfg = cfg_constrain(cfg)

    dataset_base_dir = cfg.dataset_base_dir
    dataset_name = cfg.dataset_name

    dataset_dir = osp.join(dataset_base_dir, dataset_name)
    scene_paths = [
        osp.join(dataset_dir, scene_name) for scene_name in os.listdir(dataset_dir)
    ]

    # Run only on wanted scenes
    if "scene_list" in cfg:
        if cfg.scene_list is not None:
            scene_paths = [
                scene_path
                for scene_path in scene_paths
                if osp.basename(scene_path) in cfg.scene_list
            ]
    
    if "exclude_scene_list" in cfg:
        if cfg.exclude_scene_list is not None:
            scene_paths = [
                scene_path
                for scene_path in scene_paths
                if osp.basename(scene_path) not in cfg.exclude_scene_list
            ]
    scene_paths = scene_paths[: cfg.n_scene]

    random.shuffle(scene_paths)

    if cfg.ray.enable:
        if cfg.ray.slurm:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                object_store_memory=10**10,
                num_cpus=math.ceil(cfg.ray.n_workers * cfg.ray.n_cpus_per_worker),
                num_gpus=math.ceil(cfg.ray.n_workers * cfg.ray.n_gpus_per_worker),
                local_mode=cfg.ray.local_mode,
                ignore_reinit_error=True,
            )

        logger.info(f"Use ray for eval each scene, total: {cfg.ray.n_workers} workers")
        pb = ProgressBar(len(scene_paths), "Object reconstruction begin...")
        all_subsets = chunks_balance(scene_paths, cfg.ray.n_workers)
        remote_func = eval_core_ray_wrapper.options(num_cpus=cfg.ray.n_cpus_per_worker, num_gpus=cfg.ray.n_gpus_per_worker)
        results = [
            remote_func.remote(subset, cfg, worker_id=id, pba=pb.actor)
            for id, subset in enumerate(all_subsets)
        ]
        pb.print_until_done()
        results = ray.get(results)
        # Edit(Fitz): add sfm_result_paths
        # metric_dict = dict(ChainMap(*[k for k in results]))
        metric_dict = dict(ChainMap(*[k for k, _ in results]))
        sfm_result_paths = set(*[k for _, k in results])
        ray.shutdown()
    else:
        # Edit(Fitz): add sfm_result_paths
        # metric_dict = eval_core(scene_paths, cfg)
        metric_dict, sfm_result_paths = eval_core(scene_paths, cfg)

    if not cfg.neuralsfm.close_eval:
        # Aggregate metrics from all scenes and output:
        output_base_dir = cfg.output_base_dir
        if 'phase' not in cfg or cfg.phase == 'reconstruction':
            dataset_output_dir = osp.join(output_base_dir, dataset_name)
            Path(dataset_output_dir).mkdir(exist_ok=True, parents=True)

            if cfg.method == "DetectorFreeSfM":
                file_name = "_".join(
                    [
                        "aggregrated_metrics" if cfg.use_prior_intrin else 'aggregated_metrics_no_intrin',
                        "_".join(
                            [
                                cfg.method,
                                cfg.neuralsfm.NEUSFM_coarse_matcher,
                                cfg.neuralsfm.NEUSFM_coarse_match_type,
                                f"round{cfg.neuralsfm.NEUSFM_coarse_match_round}"
                                if cfg.neuralsfm.NEUSFM_coarse_match_round is not None
                                else "",
                            ]
                        ),
                        str(cfg.exp_name)
                    ]
                ) + '.txt'
                output_path = osp.join(
                    dataset_output_dir, file_name
                )
            else:
                output_path = osp.join(
                    dataset_output_dir, f"aggregrated_metrics{'_no_intrin' if not cfg.use_prior_intrin else ''}_{cfg.method}_{cfg.exp_name}.txt"
                )
            aggregate_multi_scene_metrics(
                metric_dict,
                dataset_name=cfg.dataset_name,
                verbose=True,
                output_path=output_path,
            )
    # Edit(Fitz): add sfm_result_paths
    return sfm_result_paths

@hydra.main(config_path="hydra_configs/", config_name="base.yaml")
def main(cfg: DictConfig):
    globals()[cfg.type](cfg)


if __name__ == "__main__":
    main()
