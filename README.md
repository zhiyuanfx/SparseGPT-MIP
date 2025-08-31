# SparseGPT-MIP
A trial refinement of SparseGPT with added support for a Gurobi-based MIP mask solver.

## Dependencies
- `torch`  
- `transformers`  
- `datasets`  
- `gurobipy`  

## Features
- **Gurobi-based MIP Mask Solver**: Added support for solving mask optimization problems using Gurobi's Mixed-Integer Programming (MIP) solver.  
- **LLaMA Model Support**: Includes functionality for pruning OPT and LLaMA models.  
- **Parallelism for Gurobi WLS**: Utilizes 2-process parallelism to handle Gurobi WLS session limits, configurable in `mask_solver.py`.  

## Example Usages

### SparseGPT with MIP Solver
```bash
# n:m semi-structured pruning with Gurobi MIP solver
python sparsegpt/opt.py local_opt_directory/ wikitext2 --prunen 2 --prunem 4 --nsamples 64 --save local_pruned_opt_directory --solver mippruner

# unstructured sparsity with MIP solver
python sparsegpt/opt.py local_opt_directory/ wikitext2 --sparsity 0.5 --nsamples 64 --save local_pruned_opt_directory --solver mippruner

# unstructured sparsity with baseline SparseGPT
python sparsegpt/opt.py local_opt_directory/ wikitext2 --sparsity 0.5 --nsamples 64 --save local_pruned_opt_directory --solver sparsegpt
```

## LLaMA Model with MIP Solver
We also support LLaMA models with the same approach:
```bash
python sparsegpt/llama.py local_llama_directory/ c4 --prunen 2 --prunem 4 --nsamples 64 --solver mippruner
```

## Gurobi License
Users must provide their own Gurobi license in the gurobi_config file.
To use a software version of the license, adjust the code in the solve_single_row() method in mask_solver.py.

## Parallelism for Gurobi WLS
Due to session limits of Gurobi WLS, we use a 2-process parallelism approach.
This can be adjusted in the mask_solver.py file if needed.

## Credit
This repository is a modified version of SparseGPT (Apache 2.0 license).  
Original source: https://github.com/IST-DASLab/sparsegpt?tab=readme-ov-file
Modifications: Added Gurobi-based MIP solver, support for LLaMA pruning, 
and parallelism for Gurobi WLS.  