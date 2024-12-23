import numpy as np
import logging
import config

from utils import setup_logger
import torch.multiprocessing as mp
import loggers as lg

class Node():

	def __init__(self, state):
		self.state = state
		self.playerTurn = state.playerTurn
		self.id = state.id
		self.edges = []

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():

	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
		self.inNode = inNode
		self.outNode = outNode
		self.playerTurn = inNode.state.playerTurn
		self.action = action

		self.stats =  {
					'N': 0,
					'W': 0,
					'Q': 0,
					'P': prior,
				}
				

class MCTS():

	def __init__(self, root, cpuct):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.addNode(root)
	
	def __len__(self):
		return len(self.tree)

	def moveToLeaf(self):

		lg.logger_mcts.info('------MOVING TO LEAF------')

		breadcrumbs = []
		currentNode = self.root

		done = 0
		value = 0

		while not currentNode.isLeaf():

			lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
		
			maxQU = -99999

			if currentNode == self.root:
				epsilon = config.EPSILON
				nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
			else:
				epsilon = 0
				nu = [0] * len(currentNode.edges)

			Nb = 0
			for action, edge in currentNode.edges:
				Nb = Nb + edge.stats['N']

			for idx, (action, edge) in enumerate(currentNode.edges):

				U = self.cpuct * \
					((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
					np.sqrt(Nb) / (1 + edge.stats['N'])
					
				Q = edge.stats['Q']

				lg.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
					, action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
					, np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))

				if Q + U > maxQU:
					maxQU = Q + U
					simulationAction = action
					simulationEdge = edge

			lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)

			newState, value, done = currentNode.state.takeAction(simulationAction) #the value of the newState from the POV of the new playerTurn
			currentNode = simulationEdge.outNode
			breadcrumbs.append(simulationEdge)

		lg.logger_mcts.info('DONE...%d', done)

		return currentNode, value, done, breadcrumbs



	def backFill(self, leaf, value, breadcrumbs):
		lg.logger_mcts.info('------DOING BACKFILL------')

		currentPlayer = leaf.state.playerTurn

		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn == currentPlayer:
				direction = 1
			else:
				direction = -1

			edge.stats['N'] = edge.stats['N'] + 1
			edge.stats['W'] = edge.stats['W'] + value * direction
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

			lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
				, value * direction
				, playerTurn
				, edge.stats['N']
				, edge.stats['W']
				, edge.stats['Q']
				)

			edge.outNode.state.render(lg.logger_mcts)

	def addNode(self, node):
		self.tree[node.id] = node

	def merge_with(self, other_tree):
		"""다른 트리의 통계 정보를 현재 트리에 병합"""
		for node_id, other_node in other_tree.items():
			if node_id not in self.tree:
				self.tree[node_id] = other_node
				continue
				
			current_node = self.tree[node_id]
			# 엣지를 action을 키로 하는 딕셔너리로 변환
			current_edges = {action: edge for action, edge in current_node.edges}
			other_edges = {action: edge for action, edge in other_node.edges}
			
			# 모든 action에 대해 통계 병합
			all_actions = set(current_edges.keys()) | set(other_edges.keys())
			for action in all_actions:
				if action in current_edges and action in other_edges:
					# 양쪽 모두에 있는 엣지는 통계 합산
					current_edge = current_edges[action]
					other_edge = other_edges[action]
					current_edge.stats['N'] += other_edge.stats['N']
					current_edge.stats['W'] += other_edge.stats['W']
					if current_edge.stats['N'] > 0:
						current_edge.stats['Q'] = current_edge.stats['W'] / current_edge.stats['N']
				elif action in other_edges:
					# 새로운 엣지 추가
					current_node.edges.append((action, other_edges[action]))