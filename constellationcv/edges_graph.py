class graph_node(object):
	"""docstring for graph_node"""
	def __init__(self, x,y,neighbors):
		self.x=x
		self.y=y
		self.neighbors=neighbors
		self.visited=false

	def getNeighbors(self):
		return self.neighbors

	def getCoords(self):
		return [self.x,self.y]

	def isVisited(self):
		return self.visited

class graph(object):
	def __init__(self):
		self.nodes_list=[]

	def add_node(self, new_node):
		self.nodes_list.append(new_node)

	def findNode(self, x, y):
		for node in self.nodes_list:
			if node.getCoords==[x,y]:
				return node
		return -1
		