#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel


import hpu_util.utils as utils
import hpu_util.build as build
# from hpu_util.build import build_detection_test_loader,build_detection_train_loader
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

import detectron2.utils.comm as comm

from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
# from hpu_util.build_mdl import build_model


from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
logger = logging.getLogger("detectron2")


distributed = comm.get_world_size() > 1

def get_evaluator(cfg, dataset_name, output_folder=None):
    return COCOEvaluator(dataset_name, output_dir=output_folder)

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        logger.info(f"**chk_dataloader : {type(data_loader)}")
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results **is_main_process** for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def get_augs(cfg):
    """Add all the desired augmentations here. A list of availble augmentations
    can be found here:
       https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    """
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    # horizontal_flip: bool = cfg.INPUT.RANDOM_FLIP == "horizontal"
    # augs.append(T.RandomFlip(horizontal=horizontal_flip, vertical=not horizontal_flip))
    # # Rotate the image between -90 to 0 degrees clockwise around the centre
    # augs.append(T.RandomRotation(angle=[-90.0, 0.0]))
    return augs

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
        
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = utils.DetectionCheckpointer_hpu(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = utils.PeriodicCheckpointer_hpu(
        args,checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    logger.info(f"**chk_writers comm.is_main_process :{comm.is_main_process()}")


    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    mapper = DatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg))
    data_loader = build_detection_train_loader(cfg,mapper=mapper)
    logger.info(f"**chk_train data_loader: {type(data_loader)}******")

    logger.info("Starting training from iteration {}".format(start_iter))
    print(model.device)
    if(cfg.MODEL.DEVICE == 'hpu'):
        utils.permute_params(model, True, args.enable_lazy)
        utils.permute_momentum(optimizer, True, args.enable_lazy)

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            # for d in data:
            #     print(d)
                # print(d['image']) 
            print("device used")       
            print(model.device)
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            
            if args.enable_lazy:
                import habana_frameworks.torch.core as htcore
                htcore.mark_step()

            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                print("testing")
                # if(cfg.MODEL.DEVICE == 'hpu'):
                #     utils.permute_params(model, False, args.enable_lazy)
                #     utils.permute_momentum(optimizer, False, args.enable_lazy)
                do_test(cfg, model)
                # if(cfg.MODEL.DEVICE == 'hpu'):
                #     utils.permute_params(model, True, args.enable_lazy)
                #     utils.permute_momentum(optimizer, True, args.enable_lazy)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                print("writing")
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE ='cpu'

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code

    #  TODO: print hpu info
    return cfg

from detectron2.data.datasets import register_coco_instances

def main(args,cfg):

    register_coco_instances(
        cfg.DATASETS.TRAIN[0] ,
        {},
        args.json_annotation_train,
        args.image_path_train
    )

    register_coco_instances(
        cfg.DATASETS.TEST[0], 
        {}, 
        args.json_annotation_val, 
        args.image_path_val
    )

    # utils.init_distributed_mode(args)
    if args.enable_lazy and cfg.MODEL.DEVICE=='hpu':
        os.environ["PT_HPU_LAZY_MODE"]="1"
        import habana_frameworks.torch.core as htcore

    if cfg.MODEL.DEVICE == 'hpu':
        import habana_dataloader
        build.data_loader_type=habana_dataloader.HabanaDataLoader
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        load_habana_module()
        logger.info("**chk_dataloaded: {utils.data_loader_type}")


    model = build_model(cfg)

    logger.info("**chk_Model is on {} device".format(next(model.parameters()).device))

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        utils.DetectionCheckpointer_hpu(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return do_test(cfg, model)

    if distributed:
        if cfg.MODEL.DEVICE == 'hpu':
            model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=100, broadcast_buffers=False,
                    gradient_as_bucket_view=True)
        else:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)


    logger.info(f"**chk_distributed: {distributed} .size is {comm.get_world_size()}")


    do_train(cfg, model, resume=args.resume)

    # do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)

import sys

if __name__ == "__main__":
    parser = default_argument_parser()

    parser.add_argument(
        "--json_annotation_train",
        help="The path to the training set JSON annotation",
    )
    parser.add_argument(
        "--image_path_train",
        help="The path to the training set image folder",
    )
    parser.add_argument(
        "--json_annotation_val",
        help="The path to the validation set JSON annotation",
    )
    parser.add_argument(
        "--image_path_val",
        help="The path to the validation set image folder",
    )

    ## ref model-reference
    parser.add_argument('--dl_worker_type', 
        default='MP', type=lambda x: x.upper(),choices = ["MP", "HABANA"], 
        help='select multiprocessing or habana accelerated')

    parser.add_argument('--enable-lazy', action='store_true',
                        help='whether to enable Lazy mode, if it is not set, your code will run in Eager mode')

    parser.add_argument('--eval_only', action='store_true',
                        help='whether to enable Lazy mode, if it is not set, your code will run in Eager mode')

    args = parser.parse_args()
    # args.json_annotation_train="/home/LayoutWorkSpace/10kTable-Detection/data/train.json"
    # args.image_path_train="/home/LayoutWorkSpace/10kTable-Detection/data/"
    # args.json_annotation_val="/home/LayoutWorkSpace/10kTable-Detection/data/val.json"
    # args.image_path_val="/home/LayoutWorkSpace/10kTable-Detection/data/"
    
   

    # args.dataset_name="10kTable-layout"
    # args.json_annotation_train="/home/data/train.json"
    # args.json_annotation_val="/home/data/val.json"
    # args.image_path_train="/home/data/"
    # args.image_path_val="/home/data/"
    # args.config_file="/home/training/faster_rcnn_R_50_FPN_3x.yml"
    # # args.eval_only=False
    print("Command Line Args:", args)
    
    
    cfg = setup(args)
    
    

    if cfg.MODEL.DEVICE=='hpu':
        # utils.init_distributed_mode(args)\
        main(args,cfg)
    else:      
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,cfg,),
        )

