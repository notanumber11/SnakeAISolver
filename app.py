from flask import Flask, jsonify, request
from flask_cors import CORS

from model.game_status import GameStatus
from model.point import Point

app = Flask(__name__)
CORS(app)

snakeObj = GameStatus(0, 0, 0)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/snake/validPosition", methods=["POST"])
def is_valid_position():
    data = request.get_json()
    print(data)
    pos1 = data["pos"]
    pos = Point(pos1[0], pos1[1])
    size = data["size"]
    snake = data["snake"]
    snakePoints = []
    for el in snake:
        snakePoints.append(Point(int(el[0]), int(el[1])))
    answer = snakeObj.can_move_to(pos, snakePoints, size)
    return jsonify(answer)


if __name__ == '__main__':
    app.run(debug=True)
