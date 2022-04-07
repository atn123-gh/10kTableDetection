# The codes is adapted from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/build.py and 
# https://github.com/ember816/Model-References/tree/master/PyTorch/computer_vision,
# and https://docs.habana.ai/en/latest/PyTorch/index.html for hpu migration

import logging

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
import detectron2.utils.comm as comm
from fvcore.common.checkpoint import Checkpointer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

import torch.distributed as dist
import os
logger = logging.getLogger("detectron2")

#permute the params from filters first (KCRS) to filters last(RSCK) or vice versa.
#and permute from RSCK to KCRS is used for checkpoint saving
def permute_params(model, to_filters_last, lazy_mode):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if(param.ndim == 4):
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute((3, 2, 0, 1))  # permute RSCK to KCRS

    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()


def permute_momentum(optimizer, to_filters_last, lazy_mode):
    # Permute the momentum buffer before using for checkpoint
    for group in optimizer.param_groups:
        for p in group['params']:
            param_state = optimizer.state[p]
            if 'momentum_buffer' in param_state:
                buf = param_state['momentum_buffer']
                if(buf.ndim == 4):
                    if to_filters_last:
                        buf = buf.permute((2,3,1,0))
                    else:
                        buf = buf.permute((3,2,0,1))
                    param_state['momentum_buffer'] = buf

    if lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()


# TODO: add testcase
class DetectionCheckpointer_hpu(DetectionCheckpointer):
    """
    Implemented for hpu
    Same as :class:`DetectionCheckpointer`, but is able to:
    has getter setter for model and optimizer for permutation
    """

    def __init__(self, model, save_dir="",*, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        self._model = model
        self._optimizer= None

        for k, v in checkpointables.items():
            if k == 'optimizer':
                self._optimizer= v

        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables
        )


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._x = value


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._x = value



# TODO: add testcase
class PeriodicCheckpointer_hpu(PeriodicCheckpointer):
    """
    Implemented for hpu
    Same as :class:`PeriodicCheckpointer`, but it
    perfrom permutation before and after saving 
    https://docs.habana.ai/en/latest/Migration_Guide/Migration_Guide.html#convolution-weight-ordering-in-pytorch-habana-vision-topologies
    """


    def __init__(
        self,args,
        checkpointer: DetectionCheckpointer_hpu,
        period: int,
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
        file_prefix: str = "model",
    ) -> None:
        self.model=checkpointer.model
        self.optimizer=checkpointer.optimizer
        self.args=args
        super().__init__(checkpointer,period,max_iter,max_to_keep,file_prefix)

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Same argument as :meth:`PeriodicCheckpointer.save`.
        Use this method to perfrom permutation before and after saving.

        Args:
            name (str): file name.
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        # permute_params(self.model, False, self.args.enable_lazy)
        # permute_momentum(self.optimizer, False, self.args.enable_lazy)
        self.checkpointer.save(name, **kwargs)
        # permute_params(self.model, True, self.args.enable_lazy)
        # permute_momentum(self.optimizer, True, self.args.enable_lazy)



import torch
import torch.distributed as dist
import os
mpi_comm = None

def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
    else :
        try:
           from mpi4py import MPI
           global mpi_comm
           mpi_comm = MPI.COMM_WORLD
           size = mpi_comm.Get_size() # new: gives number of ranks in comm
           rank = mpi_comm.Get_rank()
           if size > 1:
              os.environ['MASTER_ADDR'] = 'localhost'
              os.environ['MASTER_PORT'] = '12355'
              os.environ['RANK'] = str(rank)
              os.environ['WORLD_SIZE'] = str(size)
              os.environ["ID"] = os.environ["RANK"]
              args.world_size = int(os.environ['WORLD_SIZE'])
              if 'LOCAL_RANK' not in os.environ:
                  args.local_rank = rank
              args.distributed = True
           else:
              print('Not using distributed mode')
              logger.info('Not using distributed mode')
              logger.info(f"world size {os.environ['WORLD_SIZE']}")
            

              args.distributed = False
              return
        except Exception as e:
           args.distributed = False
           print(e)
           logger.info(f"world size {os.environ['WORLD_SIZE']}")
           logger.info("**mpi4py is not available, using mpirun will not run distributed mode")
           print("**mpi4py is not available, using mpirun will not run distributed mode")
           return
    logger.info("hccl")
    logger.info(f"world size {os.environ['WORLD_SIZE']}")
    import torch.utils.data.distributed
    import torch.distributed as dist
    args.dist_backend = 'hccl'
    import habana_frameworks.torch.core.hccl
    dist._DEFAULT_FIRST_BUCKET_BYTES = 100*1024*1024
    dist.init_process_group(args.dist_backend, init_method='env://')
    os.environ["MAX_WAIT_ATTEMPTS"] = "90"
    print("world_size = {}".format(args.world_size))
    print("distributed={}".format(args.distributed))

