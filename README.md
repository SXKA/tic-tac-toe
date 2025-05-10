# tic-tac-toe
This is a terminal-based Tic‑Tac‑Toe game implemented in Python.
## Features
- **Two-Agent Training Setup**  
  Both agents play against each other, updating Q-table.
- **Custom TF‑Agents Environment**  
  Compatible with RL workflows; includes legal move masking and reward design.
- **Human vs. AI Mode**  
  Play directly in the terminal using trained Q-values.
- **Q-Table Persistence**  
  Save/load `q_table.pkl` for faster iteration or deployment.
## Usage
### Training the Agent
Run the training loop where two agents take turns and update the Q-table:
```bash
python q_learning.py
```
- Hyperparameters (learning rate, ε-greedy strategy, number of episodes) can be adjusted at the top of the file.
- A `q_table.pkl` will be saved at the end of training.
### Play Against the AI
Use the trained Q-table to play interactively:
```bash
python main.py
```
- Choose whether you want to go first.
- Enter move positions from 0 to 8 (left to right, top to bottom).
- The AI plays using greedy Q-values.
## Training Outcomes Over Episodes
<div align="center">
<img src="https://github.com/SXKA/tic-tac-toe/blob/main/png/q_learning_result.png" alt="q_learning_result"/>
</div>
