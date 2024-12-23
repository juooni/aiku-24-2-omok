from pyvirtualdisplay import Display
import tkinter as tk
from tkinter import messagebox
from game import Game  # 기존 게임 상태 로직
from agent import Agent, User
from model1 import Residual_CNN
import config

# Step 1: 가상 디스플레이 시작
display = Display(visible=0, size=(1024, 768))  # 가상 디스플레이 생성
display.start()

# Step 2: GUI 코드를 여기에 추가
class GomokuGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gomoku Game")

        # 바둑판 크기와 초기화
        self.board_size = 15
        self.cell_size = 40
        self.game = Game()  # 기존 Game 객체를 생성

        # 플레이어와 AI 설정
        self.user = User("User", self.game.state_size, self.game.action_size)
        self.ai = self.initialize_ai()

        self.canvas = tk.Canvas(
            master, width=self.board_size * self.cell_size, height=self.board_size * self.cell_size, bg="white"
        )
        self.canvas.pack()

        # 바둑판 그리기
        self.draw_board()

        # 이벤트 연결
        self.canvas.bind("<Button-1>", self.handle_click)

    def initialize_ai(self):
        """AI 초기화"""
        ai_nn = Residual_CNN(
            config.REG_CONST,
            config.LEARNING_RATE,
            self.game.input_shape,
            self.game.action_size,
            config.HIDDEN_CNN_LAYERS,
        )
        ai_nn.read("gomoku", 1, 25)  # 모델 불러오기 (파일 이름과 버전은 수정)
        return Agent("AI", self.game.state_size, self.game.action_size, config.MCTS_SIMS, config.CPUCT, ai_nn, device=None)

    def draw_board(self):
        """바둑판 그리기"""
        for i in range(self.board_size):
            self.canvas.create_line(
                self.cell_size / 2, self.cell_size * (i + 0.5), self.cell_size * self.board_size - self.cell_size / 2,
                self.cell_size * (i + 0.5), fill="black"
            )
            self.canvas.create_line(
                self.cell_size * (i + 0.5), self.cell_size / 2, self.cell_size * (i + 0.5),
                self.cell_size * self.board_size - self.cell_size / 2, fill="black"
            )

    def handle_click(self, event):
        """클릭 이벤트 처리"""
        row = int(event.y / self.cell_size)
        col = int(event.x / self.cell_size)

        if self.game.gameState.board[row, col] != 0:
            messagebox.showwarning("Invalid Move", "This cell is already occupied!")
            return

        # 유저가 돌을 놓음
        action = row * self.board_size + col
        self.update_game(action, player_turn=1)

        # AI 차례
        if not self.game.gameState.isTerminal():
            action, _, _, _ = self.ai.act(self.game.gameState, tau=0)
            self.update_game(action, player_turn=-1)

    def update_game(self, action, player_turn):
        """게임 상태 업데이트 및 GUI 갱신"""
        row, col = divmod(action, self.board_size)

        # 돌 추가
        color = "black" if player_turn == 1 else "white"
        self.canvas.create_oval(
            col * self.cell_size + 5,
            row * self.cell_size + 5,
            (col + 1) * self.cell_size - 5,
            (row + 1) * self.cell_size - 5,
            fill=color,
        )

        # 게임 로직 업데이트
        self.game.gameState, value, done, _ = self.game.step(action)

        # 게임 종료 체크
        if done:
            if value == 1:
                messagebox.showinfo("Game Over", "You win!")
            elif value == -1:
                messagebox.showinfo("Game Over", "AI wins!")
            else:
                messagebox.showinfo("Game Over", "It's a draw!")
            self.canvas.unbind("<Button-1>")  # 게임 종료 시 클릭 비활성화


if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()

# Step 3: 가상 디스플레이 종료
display.stop()
