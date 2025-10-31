# Distributed Pipeline - Multi-GPU Training

## What's Different

single gpu? nah. this version splits work across multiple gpus.

main differences from `main_pipeline.py`:
- **parallel training**: multiple processes, one per gpu
- **synchronized gradients**: all gpus share learning
- **distributed data**: population split across workers
- **collective ops**: all processes coordinate together

## Why Distributed

- **faster training**: linear speedup with more gpus (ideally)
- **bigger populations**: more memory = larger population sizes
- **parallel eval**: fitness evaluation across multiple devices
- **scale up**: same code, just add more gpus

## How It Works

### Setup Phase

```python
AlphaDesignPipeline.setup_distributed(rank, world_size)
```

- each gpu gets a unique **rank** (0 to N-1)
- **world_size** = total number of gpus
- uses NCCL backend for gpu communication
- sets `MASTER_ADDR` and `MASTER_PORT` for coordination

### Process Groups

- **rank 0** = main process. handles logging, checkpoints, etc.
- **other ranks** = worker processes. do computation, sync results.
- `dist.barrier()` keeps everyone in sync

### Model Wrapping

```python
self.neural_network = DDP(
    self.neural_network,
    device_ids=[self.rank],
    output_device=self.rank
)
```

- wraps neural network with `DistributedDataParallel` (DDP)
- each gpu gets its own copy
- gradients averaged across all copies during backprop
- `find_unused_parameters=True` for complex models(need to research a bit more)

### Data Distribution

fitness evaluation:
- each process evaluates subset of population
- `all_gather_object()` collects results to main process
- main process aggregates and broadcasts back

training:
- all processes compute gradients on their data
- `all_reduce()` averages gradients
- synchronized optimizer step

### Synchronization Points

key sync operations:
- `dist.broadcast_object_list()` - send data from rank 0 to all
- `dist.all_gather_object()` - collect data from all processes
- `dist.all_reduce()` - sum/average values across processes
- `dist.barrier()` - wait for all processes to reach this point

## Training Flow

1. **initialize**: rank 0 creates initial population
2. **broadcast**: population sent to all workers
3. **evaluate**: each worker evaluates subset
4. **gather**: fitness scores collected to rank 0
5. **evolve**: rank 0 generates next generation
6. **broadcast**: new population distributed
7. **nn training**: all workers train neural network with synchronized gradients
8. **repeat**: back to step 3

## Key Differences

### Population Management

**main_pipeline**:
```python
new_population = self.generate_next_population()
```

**distributed_pipeline**:
```python
new_population = self.generate_next_population()
new_population = self.broadcast_population(new_population)
```

### Neural Network Training

**main_pipeline**:
```python
loss.backward()
optimizer.step()
```

**distributed_pipeline**:
```python
loss.backward()  # DDP handles gradient synchronization
optimizer.step()  # all processes step together
```

### Logging

**main_pipeline**:
```python
self.logger.info("training...")
```

**distributed_pipeline**:
```python
if self.is_main_process:
    self.logger.info("training...")
```

only rank 0 logs. this avvoids spam from all processes.

## Usage

### Single GPU (fallback)
```python
pipeline = AlphaDesignPipeline("config.json", rank=-1, world_size=1)
results = pipeline.run_complete_pipeline(base_params)
```

### Multi-GPU
```python
main_distributed("config.json", world_size=2)
```

uses `torch.multiprocessing.spawn()` to launch processes:
```python
mp.spawn(
    run_distributed_training,
    args=(world_size, config_path, base_params),
    nprocs=world_size,
    join=True
)
```

### Environment Variables

```bash
export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
```

for multi-node (not implemented yet):
```bash
export MASTER_ADDR='192.168.1.1'
export MASTER_PORT='12355'
export WORLD_SIZE=8 
export RANK=0 
```

## Memory Management

each gpu has its own:
- copy of neural network
- subset of population for evaluation
- local gradients

shared across gpus:
- synchronized model weights
- aggregated fitness scores
- generation results (on rank 0)


## Limitations

current implementation:
- only supports single-node multi-gpu
- fitness evaluation not fully parallelized (todo)
- population size should be divisible by world_size
- requires all gpus to have same memory

## Cleanup

```python
AlphaDesignPipeline.cleanup_distributed()
```

destroys process group when done. call this before exit.


## References

- PyTorch DDP docs: https://pytorch.org/docs/stable/notes/ddp.html
- torch.distributed: https://pytorch.org/docs/stable/distributed.html
- NCCL backend: https://docs.nvidia.com/deeplearning/nccl/


