# Solve Sudoku 
1. Install the dependencies, perform the following, 
  ```bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install python3-venv build-essential python3-dev git-all libgl1-mesa-dev ffmpeg
  ```
2. Create a virtual environment
 ```bash
    python3 -m venv sudoku_env
 ```
3. Activate the environment
```bash
    source sudoku_env/bin/activate
 ```
4. Install the packages

```bash
    python -m pip install --upgrade pip 
    pip install -r requirements.txt
 ```
5. Launch the application to solve Sudoku
```bash
    python solve.py
 ```
