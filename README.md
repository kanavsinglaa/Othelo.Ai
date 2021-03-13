# Othelo.Ai
This Ai uses alpha beta prunning and smart heuristics to beat other agents in the game of Othelo.
#### Please drop a star if you find this helpful or atleast mildly exciting ;)
###### Note: This project is based on one used in Columbia University’s Artificial Intelligence Course (COMS W4701). Special thanks to Dr. Daniel Bauer, who developed the starter code that's further extended.

## Results
The following is the result of our smart heuristic agent using alpha beta prunning against random and basic heuristic agents:

Against a Random Moves Agent (white) | Against an Ai Agent (white)
:------------:|:--------------------:
![](gifs/randy.gif))|![](gifs/agent.gif)

## Running
To play against my Ai, run:
```bash
    python othello_gui.py -d 8 -a agent_smarter.py -l 6 -o 
```

