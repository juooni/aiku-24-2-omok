import numpy as np
import geumsu
import logging

from config import BOARD_SIZE

class Game:

	def __init__(self):	
		self.currentPlayer = 1
		self.gameState = GameState(np.array([0]*(BOARD_SIZE ** 2), dtype=np.int64), 1)
		self.actionSpace = np.array([0]*(BOARD_SIZE ** 2), dtype=np.int64)
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.grid_shape = (BOARD_SIZE,BOARD_SIZE)
		self.input_shape = (2,BOARD_SIZE,BOARD_SIZE)
		self.name = 'gomoku'
		self.state_size = len(self.gameState.binary)
		self.action_size = len(self.actionSpace)

	def reset(self):
		self.gameState = GameState(np.array([0]*(BOARD_SIZE ** 2), dtype=np.int64), 1)
		self.currentPlayer = 1
		return self.gameState

	def step(self, action):
		next_state, value, done = self.gameState.takeAction(action)
		self.gameState = next_state
		# print(self.gameState.printBoard())
		self.currentPlayer = -self.currentPlayer
		info = None
		return ((next_state, value, done, info))

	def identities(self, state, actionValues):
		identities = [(state,actionValues)]

		currentBoard = state.board
		currentAV = actionValues

		currentBoard = np.array([currentBoard[j] for i in range(0, (BOARD_SIZE ** 2), BOARD_SIZE) for j in range(i+(BOARD_SIZE - 1), i-1, -1)])
		currentAV = np.array([currentAV[j] for i in range(0, (BOARD_SIZE ** 2), BOARD_SIZE) for j in range(i+(BOARD_SIZE - 1), i-1, -1)])

		identities.append((GameState(currentBoard, state.playerTurn), currentAV))

		return identities


class GameState():
	def __init__(self, board, playerTurn):
		self.board = board
		self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.winners = self._winners()
		self.playerTurn = playerTurn
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions = self._allowedActions()
		self.isEndGame = self._checkForEndGame()
		self.value = self._getValue()
		self.score = self._getScore()

	def _winners(self):
		winners = []
		rows, cols, mok = BOARD_SIZE, BOARD_SIZE, 5

		for r in range(rows):  # 가로
			for c in range(0, cols - mok + 1):
				idx = r*cols + c
				winners.append([idx + k for k in range(mok)])
				
		for r in range(0, rows - mok + 1):  # 세로
			for c in range(cols):
				idx = r*cols + c
				winners.append([idx + k * cols for k in range(mok)])

		for r in range(0, rows - mok + 1):  # 대각선(우하향)
			for c in range(0, cols - mok + 1):
				idx = r*cols + c
				winners.append([idx + k * (cols + 1) for k in range(mok)])

		for r in range(mok - 1, rows):  # 대각선(우상향)
			for c in range(0, cols - mok + 1):
				idx = r*cols + c
				winners.append([idx - k * (cols - 1) for k in range(mok)])

		return winners

	def _allowedActions(self):
		allowed = []

		# 오목알이 없고 금수가 아닐 경우 allowed
		for i in range(len(self.board)):
			if not self.board[i] and geumsu.is_allowed(self.board, i):
				allowed.append(i)

		return allowed

	def _binary(self):
		currentplayer_position = np.zeros(len(self.board), dtype=np.int64)
		currentplayer_position[self.board==self.playerTurn] = 1

		other_position = np.zeros(len(self.board), dtype=np.int64)
		other_position[self.board==-self.playerTurn] = 1

		position = np.append(currentplayer_position,other_position)

		return (position)

	def _convertStateToId(self):
		player1_position = np.zeros(len(self.board), dtype=np.int64)
		player1_position[self.board==1] = 1

		other_position = np.zeros(len(self.board), dtype=np.int64)
		other_position[self.board==-1] = 1

		position = np.append(player1_position,other_position)

		id = ''.join(map(str,position))

		return id

	def _checkForEndGame(self):
		if not self.allowedActions:
			return 1

		for x,y,z,a,b in self.winners:
			if (self.board[x] + self.board[y] + self.board[z] + self.board[a] + self.board[b] == 5 * -self.playerTurn):
				return 1
		return 0


	def _getValue(self):
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose
		for x,y,z,a,b in self.winners:
			if (self.board[x] + self.board[y] + self.board[z] + self.board[a] + self.board[b] == 5 * -self.playerTurn):
				return (-1, -1, 1)
		return (0, 0, 0)


	def _getScore(self):
		tmp = self.value
		return (tmp[1], tmp[2])




	def takeAction(self, action):
		newBoard = np.array(self.board)
		newBoard[action]=self.playerTurn
		
		newState = GameState(newBoard, -self.playerTurn)

		value = 0
		done = 0

		if newState.isEndGame:
			value = newState.value[0]
			done = 1

		return (newState, value, done) 




	def render(self, logger):
		for r in range(BOARD_SIZE):
			logger.info([self.pieces[str(x)] for x in self.board[BOARD_SIZE*r : (BOARD_SIZE*r + BOARD_SIZE)]])
		logger.info('--------------')
	
	def printBoard(self):
		for r in range(BOARD_SIZE):
			print(*[self.pieces[str(x)] for x in self.board[BOARD_SIZE*r : (BOARD_SIZE*r + BOARD_SIZE)]])
		print('--------------')
