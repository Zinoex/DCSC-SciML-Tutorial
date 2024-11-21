# DCSC-SciML-Tutorial
Code to accompany the presentation at DCSC SciML Tutorial 22/11/2024.


## Preparation
Before the tutorial, please have completed the following list of items. 

1. Clone this repository and open a terminal in the root folder of the repo.
2. Run `julia --project=.` - this will start Julia with the environment configured
   to the current folder (and by extension use the `Project.toml` file in this folder
   to determine dependencies).
3. Press `]` to enter into the `Pkg` mode of the REPL and run the following three commands in order:
   1. `resolve` - will find the newest compatible set of dependencies according to `Project.toml`.
   2. `instantiate` - download the dependencies (will automatically `resolve` if no `Manifest.toml` is present).
   3. `precompile` - run the precompilation on all dependencies. Not strictly necessary since it will automatically precompile upon first execution, but running the command means we can skip the precompilation and speed up the tutorial.