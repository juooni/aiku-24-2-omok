from flask import Flask, render_template, jsonify, request
from game import Game
from agent import Agent, User
from model1 import Residual_CNN
import config
import torch

app = Flask(__name__)

# 게임 초기화
game = Game()  # Game 객체 생성
board_size = int(game.state_size ** 0.5)  # 보드 크기 계산

game.reset()  # 초기 상태로 게임 리셋
user = User("User", game.state_size, game.action_size)  # 유저 플레이어 초기화

# AI 설정
ai_nn = Residual_CNN(
    config.REG_CONST,
    config.LEARNING_RATE,
    game.input_shape,
    game.action_size,
    config.HIDDEN_CNN_LAYERS,
)

# 모델 경로를 설정하고 로드
model_path = "/home/aikusrv01/omok/torch_omok_test/DeepReinforcementLearning/models/gomokucurr_0025.pt"
ai_nn.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # CPU에서 모델 로드

# Agent 초기화
ai = Agent("AI", game.state_size, game.action_size, config.MCTS_SIMS, config.CPUCT, ai_nn, device=None)

@app.route("/")
def index():
    """기본 페이지 렌더링"""
    return render_template("index.html")

@app.route("/move", methods=["POST"])
def make_move():
    """유저의 이동 처리"""
    data = request.json
    row, col = data.get("row"), data.get("col")
    action = row * board_size + col

    # 유저의 돌 두기
    if game.gameState.board[row * board_size + col] != 0:
        return jsonify({"error": "Cell is already occupied"}), 400

    game.step(action)  # 유저가 선택한 위치를 게임 상태에 반영

    if game.gameState.isTerminal():
        return jsonify({"winner": "User", "board": game.gameState.board.tolist()})

    # AI의 돌 두기
    ai_action, _, _, _ = ai.act(game.gameState, tau=0)  # AI가 행동을 선택
    game.step(ai_action)  # AI의 행동을 게임 상태에 반영

    if game.gameState.isTerminal():
        return jsonify({"winner": "AI", "board": game.gameState.board.tolist()})

    return jsonify({"board": game.gameState.board.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
