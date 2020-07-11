## SnakeAISolver - Solving the snake game using different techniques

### What is about?
The goal of this project is to complete the snake game.

![Intro](readme_assets/board_36/intro.gif?raw=true "Title")

The following techniques/solvers have been used:
1. Random movement
1. Survival random movement
1. Depth first search algorithm
1. Reward based DNN
1. Genetic algorithm
1. Genetic algorithm with fallback to reward based dnn
1. Hamilton path

Following is a detailed explanation of each method and the associated results.
We will compare all solvers using a 36 board that starts with a snake of length 2.
The measures displayed are the result from running each of the techniques/solvers across 100 games.

#### 1. Random movement
The simplest one, the snake just takes a random decision about where to go next. As you might expect the results are quite poor.

![Random movement results](readme_assets/board_36/plot_random_solver.png?raw=true "Title")

##### Best game out of 100 for Random movement solver
![Random movement game](readme_assets/board_36/game_36_random_solver.gif?raw=true "Title")

#### 2. Survival random movement
The snake takes a random direction among the directions that will allow the snake to survive. In case there is not such direction the snake just takes a
random one and dies. As we have added another input (survive) to the solver the results are slightly better than with Random movement solver. The snake can only die with this method if 
it locks itself with its body.

![Survival Random movement results](readme_assets/board_36/plot_survival_random_solver.png?raw=true "Title")

##### Best game out of 100 for survival random movement solver
![Random movement game](readme_assets/board_36/game_36_survival_random_solver.gif?raw=true "Title")

#### 3. Depth-first search algorithm
Snake uses depth-first search algorithm to find the way to the next apple, after it finds the apple it uses DFS again to find the next one.
The snake can only die with this method if it locks itself catching an apple. The results are better than with survival random movement since in this case the snake can only lock itself
after catching an apple but not doing random explorations. The computational complexity of this method is O(n^4) being n the board size.

![DFS results](readme_assets/board_36/plot_dfs_solver.png?raw=true "Title")

##### Best game out of 100 for dfs solver
![DFS solver](readme_assets/board_36/game_36_dfs_solver.gif?raw=true "Title")

#### 4. Reward based DNN
Using as input the results from the random movement solver and a reward system the DNN is able to learn how to play snake.
The reward system is based on:

| Movement        | Reward         |
| :-------------: |:-------------:|
| Eat apple      | 0.7 |
| Die      | -1 |
| Get closer to apple      | 0.1 |
| Get further from apple      | -0.1 |

The results of this method are better than DFS but still the snake usually lose when locking itself with its own body.

![Reward Based dnn results](readme_assets/board_36/plot_reward_dnn_solver.png?raw=true "Title")

##### Best game out of 100 for reward based dnn solver
![dnn game](readme_assets/board_36/game_36_dnn_solver.gif?raw=true "Title")

#### 5. Genetic algorithm
Using as input the distance to the apple, the distance to their own body and the distance to the walls the snake is able
to learn how to beat the snake game using a genetic algorithm. The snake is able to look into 8 different directions for each of the attributes described 
previously.
The data shows that the snake is either able to finish the game or just die with low size. This is due to the fact that the genetic algorithm sometimes goes into a loop where the snake is just
moving around without taking the apple forever.

![Distance Vision Genetic solver](readme_assets/board_36/plot_distance_vision_genetic_solver.png?raw=true "Title")

##### Best game out of 100 for genetic algorithm
![Random movement game](readme_assets/board_36/game_36_distance_vision_genetic_solver.gif?raw=true "Title")

#### 6. Genetic algorithm with fallback to reward based dnn
Analyzing the data from the previous genetic algorithm it looks that it has 2 performance options: Die with really short snake length or complete the game.
In order to overcome this one option is to use the genetic algorithm until it suggest a direction that will kill the snake or until it will be stack in a loop.
If one of those situations occur the fallback reward based dnn solver will be used to find the next apple. Thanks to this trick the results are significantly better:

![Distance vision genetic solver with fallback to dnn](readme_assets/board_36/plot_distance_vision_genetic_with_fallback.png?raw=true "Title")

##### Best game out of 100 for genetic algorithm with fallback
![Random movement game](readme_assets/board_36/game_36_genetic_with_fallback_solver.gif?raw=true "Title")

#### 7. Hamilton path
The algorithm tries to find a path from the head of the snake to the tail going through all the positions of the board. After finding it the snake just
repeat this pattern again and again until finishing the game.

The disadvantages of this method are:
 - Computational cost of O(4^n) being n the number of positions of the board.
 - It is not always possible to find a halmiton path. (It is only possible to find if the board size is even)
 
Advantages:
- If a hamilton path exists, it solves the game 100% of the times.

![Hamilton solver](readme_assets/board_36/plot_hamilton_solver.png?raw=true "Title")

##### Best game out of 100 for hamilton
![Random movement game](readme_assets/board_36/game_36_hamilton_solver.gif?raw=true "Title")

#### Limitations on different board sizes
- Random movement, Ramdom Survival Movement and RewardBasedDnn solvers have not strong limits on the board size.
- DFS solver and hamilton solver computational complexity is O(4^n) being n the board size.
- Moreover hamilton solver can not find a hamilton path if the board size is odd.
- Genetic solver was trained using distances to the snake body, apple and snake. Even if those distances were
normalized using different board size affects the genetic algorithm. Slight changes on the size (6, 8, 10, 12, 16) do not 
hurt dramatically the performance of the algorithm. Moreover it is possible to continue the training of the model for new sizes
and it shows results quite fast, meaning that the model was able to learn how to beat the snake game. One limitation for
the genetic algorithm is that it does not work with odd board sizes, this can be due to genetic algorithm learning a similar
algorithm as "hamilton path" for solving the snake.

#### Gif sample of 30 seconds of genetic algorithm with 324 board size ([Click here for full video](https://www.youtube.com/watch?v=JuZHMrEsS5M))
![Gif sample of 30 seconds](readme_assets/324_board_demo.gif?raw=true "Title")
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
- Train advance genetic algorithm
   ```bash
    python main.py train_advanced_genetic
    ``` 
  
#### References
- [Slitherin snake](https://github.com/gsurma/slitherin)
- [Snake AI](https://github.com/Chrispresso/SnakeAI)