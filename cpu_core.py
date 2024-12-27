import multiprocessing
import torch

# Check for dataloader_num_workers optimal number
print(multiprocessing.cpu_count())
print(torch.cuda.is_available())