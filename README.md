# l2rpn
Solutions to the L2RPN challange: [Learning to Run a Power Network, NeurIPS 2020](https://www.public.asu.edu/~yweng2/Tutorial5/L2RPN.html)

# Learning to Run a Power Network Challenge

## In a nutshell
Below is an overview of the aim of the challenge, and the approach the designed solution uses.
For full details, see the [full challenge description paper](https://l2rpn.chalearn.org/#h.p_bRtGtZZNiiar). It also describes the environment observation and action spaces, and transition operation.

### Problem Background
With the advent of renewable energy, electric mobility, and limitations placed on engaging in new grid infrastructure projects, the task of controlling existing grids is becoming increasingly difficult, forcing grid operators to do “more with less”. This challenge aims at testing the potential of AI to address this important real-world problem for our future.

### Overview
The solution creates an agent that operates a powergrid. More specifically, the agent needs to survive (avoid blackout) for the longest number of time steps possible. 

For the robustness track, which these solutions aimed at, the agent should be robust to unexpected events and keep delivering reliable electricity everywhere even in difficult circumstances. Adversarial attacks are made to the grid lines everyday, at different times, and the agent has to learn to overcome the attacks and operate the grid safely.

## Setup

1. Install the environment
`pip install grid2op`

 - Switch the currect directory to one of the solutions `cd {solution_dir} # e.g ppo`

2. Install requirements for each solution (In each solution directory)
```
    pip install -r requirements.txt
```

3. Train the agent
```
python3 main.py
```

4. Evaluate the agent on the submission script
```
python3 eval_submission/run_check {solution_dir} 
```
 - There is an example for this in `eval_submission/example_submission/`

