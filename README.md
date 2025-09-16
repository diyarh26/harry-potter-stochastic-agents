# Harry Potter — Stochastic Agents

This project extends the Harry Potter grid world with stochastic elements.  
Wizards must maximize their score by destroying horcruxes and avoiding death eaters under uncertainty.

## Features
- Stochastic death eater movement (stay, forward, backward with equal probability)
- Horcruxes that randomly change location each turn
- Point-based system:
  - +2 points for destroying a horcrux
  - -1 point per wizard encountering a death eater
  - -2 points for resetting the environment
- New actions: `reset`, `terminate`
- Two agents implemented:
  - **WizardAgent** — baseline agent
  - **OptimalWizardAgent** — aims to act optimally

## Project Structure
- `ex3.py` — your implementation (agents and logic)
- `check.py` — environment runner
- `utils.py` — helper functions
- `inputs.py` — example input cases

## Requirements
- Python 3.10+
- No external libraries (standard library only)

## How to Run
```bash
python check.py
