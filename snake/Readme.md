### Snake Game - Traditional programming vs IA

### The following commands are intended to be run from the snake directory.

### Install dependencies
``` pip install -r requirements.txt ```
Install Tkinter if you need GUI.

### Run game
````python main.py game````

### Train basic dnn
````python main.py train_basic_dnn````

### Train basic genetic
````python main.py train_basic_genetic````

### Train advance genetic algorithm
````python main.py train_advanced_genetic````

### Run tests
````python -m unittest discover -s tests -t . ````

### Docker cheatsheet
```
docker build -t snake_ai -f Dockerfile.txt .
```

```
docker run snake_ai
```
```
docker ps -q -l
```
```
docker exec -it ba5cb87d16ca  bash
```
```
docker cp ba5cb87d16ca:/opt/ml/code/ .
```
```
docker system prune
```