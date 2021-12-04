import numpy as np
from abc import abstractmethod
from sklearn.ensemble import RandomForestClassifier

class DaggerAgent:
	def __init__(self,):
		pass

	@abstractmethod
	def select_action(self, ob):
		pass


# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
	def __init__(self, necessary_parameters=None):
		super(DaggerAgent, self).__init__()
		# init your model, use Random Forest
		self.model = RandomForestClassifier(n_estimators=120)

	# train your model with labeled data
	def update(self, data_batch, label_batch):
		# clf = RandomForestClassifier(n_estimators=10)
		self.model.fit(data_batch, label_batch)

	# select actions by your model
	def select_action(self, data_batch):
		label_predict = self.model.predict([data_batch])
		action = int(label_predict[0])
		return action





