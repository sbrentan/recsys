import csv

class IOManager:

	_path = ""

	def __init__(self, datasets_path = "../datasets/"):
		self._path = datasets_path

	def input(self, dataset_name = "", skip_first_row = False):
		with open(self._path + dataset_name, newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
			result = []
			columns = []
			for ind, row in enumerate(spamreader):
				if(ind == 0 and skip_first_row):
					columns = row[0].split(",")
				else:
					result.append(row[0].split(","))
			return columns, result
		return [], []
