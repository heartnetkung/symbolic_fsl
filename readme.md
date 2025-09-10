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

## Types of Experts
Since our blackboard system consists of many experts working at different stages of problem solving, these are different categories of experts along with their descriptions.
- __parser__ parse shapes from image with various configurations.
- __reparser__ heterogeneous graphical transformation applied to resolve parsing issues like partially-visible shapes or adjacent same-colored shapes.
- __attention-based__ homogeneous transformations applied uniformly to multiple images at once. Since a puzzle usually involves multiple different homogeneous transformations, an attention is required to cluster related shapes together as well as pair them with example outputs. These are the key problem-solving experts since the objective of the puzzles is to find these transformations and apply it to new images.
	- __create__ create new shapes of known types using the information from attending shapes.
	- __edit__ edit a column of attending shapes to be more similar to the output samples. Each shape in the editing column must be unique.
	- __specialized__ niche transformation that requires non-tabular input data and custom attention.
- __post_loop__ postprocessing operations to be done after most problem solving, including converting shapes back to image. It is triggered when the current state looks like a correct solution.