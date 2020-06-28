## SnakeAISolver - Solving the snake game using different techniques

### What is about?
The goal of this project is to solve the snake game.

<img src="https://media.giphy.com/media/QWjI4IuhD8vLr2A6UB/giphy.gif"  />

The following techniques have been used:
1. Random movement
1. Survival random movement
1. Depth first search algorithm
1. Regression DNN
1. Genetic algorithm
1. Hamilton path

Following is a detailed explanation of each method.

#### 1. Random movement
The simplest one, the snake just takes a random decision about where to go next.

#### 2. Survival random movement
Similar to the previous one but with a slightly important difference. The snake takes a random direction
among the directions that will not kill itself. In case there is not such direction the snake just takes a
random one and dies.

#### 3. Depth-first search algorithm
Snakes uses depth-first search algorithm to find the way to the next apple.

#### 4. Regression DNN
Using as input the results from the random movements and a reward system a DNN is able to learn how to play snake.

#### 5. Genetic algorithm
Using as input the distance to the apple, the distance to their own body and the distance to the walls the snake is able
to learn how to beat the snake game.

### How to use
- Install dependencies
    ```bash
     pip install -r requirements.txt
    ```
   If your python distribution does not include Tkinter you should install it for GUI support.
- Run tests
    ```bash
    python -m unittest discover -s tests -t .
    ```
- Run the snake game with different solvers:
    ```bash
    python main.py game
    ```
- Train dnn
    ```bash
    python main.py train_basic_dnn
    ```
- Train basic genetic algorithm
   ```bash
    python main.py train_basic_genetic
    ``` 
- Train basic genetic algorithm
   ```bash
    python main.py train_advanced_genetic
    ``` 
  
#### References
- [Slitherin snake](https://github.com/gsurma/slitherin)
- [Snake AI](https://github.com/Chrispresso/SnakeAI)