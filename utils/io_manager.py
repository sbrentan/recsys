import csv

class IOManager:

	_dataset_path = ""

	def __init__(self, datasets_path = "../datasets/"):
		self._dataset_path = datasets_path

	def input(self, dataset_name = "", skip_first_row = False):
		print(dataset_name)
		with open(self._dataset_path + dataset_name, newline='', encoding="utf-8") as csvfile:
			reader = csv.reader(csvfile, delimiter='\n', quotechar='|')
			result = []
			columns = []
			for ind, row in enumerate(reader):
				if(ind == 0 and skip_first_row):
					columns = row[0].split(",")
				else:
					result.append(row[0].split(","))
			return columns, result
		return [], []

	def output(self, dataset_name = "", rows = []):
		with open(self._dataset_path + dataset_name, 'w', newline='', encoding="utf-8") as csvfile:
			writer = csv.writer(csvfile, delimiter='\n', quotechar='|')
			writer.writerow(rows)


	@property
	def dataset_path(self):
		return self._dataset_path
	

