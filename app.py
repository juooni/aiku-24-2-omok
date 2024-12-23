from flask import Flask, render_template, jsonify, request
from game import Game
from agent import Agent, User
from model1 import Residual_CNN
import config
import torch
import sys

app = Flask(__name__)

# 게임 초기화
game = Game()
board_size = int(game.state_size ** 0.5)
game.reset()

# AI 설정
ai_nn = Residual_CNN(
    config.REG_CONST,
    config.LEARNING_RATE,
    game.input_shape,
    game.action_size,
    config.HIDDEN_CNN_LAYERS,
)
model_path = "/home/aikusrv01/omok/torch_omok_test/DeepReinforcementLearning/models/gomokucurr_0025.pt"
ai_nn.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
ai_nn.eval()

ai = Agent("AI", game.state_size, game.action_size, config.MCTS_SIMS, config.CPUCT, ai_nn, device=None)

@app.route("/")
def index():
    """기본 페이지 렌더링"""
    return render_template("index.html")

@app.route("/move", methods=["POST"])
def make_move():
    """유저와 AI의 턴 처리"""
    data = request.json
    row, col = data.get("row"), data.get("col")
    action = row * board_size + col

    # 유저의 턴
    if game.gameState.board[action] != 0:
        return jsonify({"error": "Cell is already occupied"}), 400

    game.step(action)  # 유저의 행동 처리

    if game.gameState.isEndGame:
        return jsonify({"winner": "User", "board": game.gameState.board.tolist()})

    # AI의 턴
    ai_action, _, _, _ = ai.act(game.gameState, tau=0)
    game.step(ai_action)

    if game.gameState.isEndGame:
        return jsonify({"winner": "AI", "board": game.gameState.board.tolist()})

    return jsonify({"board": game.gameState.board.tolist()})

if __name__ == "__main__":
    # 기본 포트는 5000, 명령줄 인자에서 포트 번호를 읽음
    port = 5000
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            port = int(sys.argv[port_index])

    app.run(debug=True, port=port)
