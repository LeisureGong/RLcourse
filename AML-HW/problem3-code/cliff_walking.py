def fall_into_cliff(row, col):
	if(row == 3):
		if(col > 0 and col < 11):
			return 1
	else:
		return 0

class Env:

	def __init__(self, row, col):
		self.pos_row = row
		self.pos_col = col
		self.state = self.pos_row * 12 + self.pos_col

	def transition(self, action):
		if(action < 2):
			if(action == 0):
				self.pos_row = self.pos_row - 1 if self.pos_row > 0 else self.pos_row
			else:
				self.pos_row = self.pos_row + 1 if self.pos_row < 3 else self.pos_row
		else:
			if(action == 2):
				self.pos_col = self.pos_col - 1 if self.pos_col > 0 else self.pos_col
			else:
				self.pos_col = self.pos_col + 1 if self.pos_col < 11 else self.pos_col

		if(fall_into_cliff(self.pos_row, self.pos_col)):
			self.reset()
			return -100

		self.state = self.pos_row * 12 + self.pos_col
		return -1

	def reset(self):
		self.pos_row = 3
		self.pos_col = 0
		self.state = self.pos_row * 12 + self.pos_col
