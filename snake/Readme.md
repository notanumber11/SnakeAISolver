### Snake Game - Traditional programming vs IA

### Run game
````python C:/Users/Denis/Desktop/SnakePython/snake/main.py````

### Run genetic algorithm training    
````python C:/Users/Denis/Desktop/SnakePython/snake/main_genetic_algorithm.py````

### Run tests
````python -m unittest discover -s C:/Users/Denis/Desktop/SnakePython/snake/tests -t C:\Users\Denis\Desktop\SnakePython\snake ````

### Docker cheatsheet
```
docker build -t snake -f Dockerfile.txt .
```

```
docker run snake
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