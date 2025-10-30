# Symbolic Few-shot Learning Algorithm

This repo stores the code for our paper named "Solving ARC-AGI with symbolic few-shot learning and tiny-scale ml". The key idea is to emulate how LLMs and humans learn few-shot learning using programming and traditional ML paradigm. This means that the algorithm can learn from a few examples and flexibly adapt the pre-existing knowledge/skills to solve the task at hand with the added benefit of being symbolic (interpretibility, explainability, robustness, maintainability, etc). The highlight of our approach is the newly propoed machine learning model that is robustly trainable from 2-3 samples! This extreme sample efficiency is obtained by searching the model space to find the simplest one that perfectly fit the data using Mixed-Integer Linear Programming and greedy matching algorithm.

The paper can be accessed from `paper/paper.pdf` folder in this repo.

## Paper Abstract

Few-shot learning is the capability of Large Language Model (LLM) that allows it to perform any task specified by a few examples. It conceptually involves the use of pre-existing knowledge stored in the model and adapt them to the task at hand. Due to its connectionist nature, its safety and explainability cannot be guaranteed. Thus, we propose an alternative symbolic approach with equivalent ability that consists of 3 parts: parameterized functions, program synthesis algorithm, and tiny-scale ML. First, parameterized functions are used to encapsulate the pre-existing knowledge. Second, the program synthesis algorithm assembles a number of functions and their parameters into programs that can perform the task at hand. Last, our novel sample-efficient ML models are trained on the given examples to predict the functions' parameters suitable for the task. To make a supervised learning model operates robustly at a 2-3 sample scale, our approach uses Occam's Razor heuristic to search for the simplest models that fit the data instead of optimizing for loss function. That is, the space of possible models is defined as a Mathematical formula and that formula is optimized to find the simplest solution with the constraints of dataset matching. In addition, thanks to the optimization being primarily performed by Mixed-Integer Linear Programming, the solution has the optimality guarantee regardless of the sample size limitation. To measure the efficacy of our approach, the benchmark is performed on ARC-AGI-1 dataset which contains a number of IQ-test-like puzzles for our system to solve. The experimental result shows that our system can solve 204 of those puzzles which means that it can recombine the predefined knowledge and adapt them in a variety of ways to perform the task. Our contribution in this work is to propose a symbolic alternative to few-show learning as well as a novel sample-efficiency technique for supervised learning. The source code is publicly available at github.com/heartnetkung/symbolic_fsl.

## Useful Command Lines
```bash
# solve all problems from #0 to #49
python3 -m arc.script.solve_range 0 50
# solve problem 30 with logging
python3 -m arc.script.solve_one 30 >temp
# solve problem 30 with full logging
python3 -m arc.script.solve_one 30 train_v1 debug >temp
# run test case with specific pattern filter
pytest -k pattern -log-level=INFO
# visualize problem 30
python3 -m arc.script.visualize 30
# solve previously solved problem for sanity checking
python3 -m arc.script.solve_previous
# check for unused functions
vulture arc
```

## Architecture
![diagram](arc_modules.png)

[source](https://app.diagrams.net/#G1FAJC1FLoCjnnSrLk9KZ1jgJZ77U93Unu#%7B%22pageId%22%3A%225f0bae14-7c28-e335-631c-24af17079c00%22%7D)

## Resources
- Reproducible experiment on [Google Colab](https://colab.research.google.com/drive/1hlM8jEvLyLtXYO2WUWbdUlYOx_33MpCc) from the paper.
- Readable solutions for the 204 problems solved by our algorithm. See folder `data/solution_1.0` in this repo.

## Citation
waiting for Arxiv endorsement...
