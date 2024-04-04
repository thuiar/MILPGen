# MILPGen

This is the official code implementation of paper "Learning to Generate Scalable MILP instances."

# Environment Setup

```bash
conda env create -f scripts/environment.yml
conda activate MILPGen
```

# Code

There are two stages of the code: the first one is instance clustering, and the second one is instance generating.

## Instance Clustering

Create directory and generating benchmark problems:
```bash
mkdir bipartite_graph
mkdir tmp_result
python 01.py --problem_type=IS --number=20 --output_dir=bipartite_graph/
python 01.py --problem_type=MVC --number=20 --output_dir=bipartite_graph/
python 01.py --problem_type=CAT --number=20 --output_dir=bipartite_graph/
```

Train VGAE to get problem embeddings and clustering instances:
```bash
python 02.py --input_dir=bipartite_graph/ --epoch=10 --output_file=tmp/02

python 03.py --class_num=3 --input_file=tmp/02 --output_file=tmp/03
```

The code `solve_gurobi.py` and `solve_scip.py` provides an entry point to solve dumped bipartite representation of MILP instances.

## Instance Generating

Instance Generating is intend for single category of problems.

We refer to scripts under `scripts/` directory.

(Scripts are under construction)

# Citations

If this work is helpful, or you want to use the codes and results in this repo, please cite the following papers:

(To be appeared)







