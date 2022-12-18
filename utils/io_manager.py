import csv
import numpy as np
from scipy.sparse import csr_matrix

class IOManager:

	_dataset_path = ""

	def __init__(self, datasets_path = "../datasets/", skip_first_row = True):
		self._dataset_path = datasets_path

	def input_csr_matrix(self, dataset_name = ""):

		# rows, columns = 111, 1001
		# matrix = sparse.lil_matrix( (rows, columns) )

		columns = None
		datas = []
		rows = []
		cols = []
		csvreader = csv.reader(open(self._dataset_path + dataset_name))
		row = 0
		for line in csvreader:
			# row, column = map(int, line)
			# matrix.data[row].append(column)
			if(columns == None):
				columns = line
			else:
				for col in range(len(line)-1):
					if(line[col+1]):
						rows.append(row-1)
						cols.append(col)
						datas.append(round(int(line[col+1])/100, 2))
			row += 1
		# print(matrix.data)
		print(len(rows), len(cols), len(datas))
		print(max(rows), max(cols))
		shape = (max(rows)+1, max(cols)+1)
		matrix = csr_matrix((datas, (rows, cols)), shape=shape, dtype=np.float32)
		# print(matrix.shape)
		return columns, matrix

	# def _insert_mcluster(indexes):


	def input_movies(self, dataset_name = ""):
		movies, mcounter = {}, 0
		features, nfeatures, fcounter = {}, None, None
		mclusters = {}
		with open(self._dataset_path + dataset_name, 'rt', encoding="utf-8") as csvfile:
			csv_reader  = csv.DictReader(csvfile, delimiter='\n', quotechar='|')
			for row in csv_reader :
				# print(mclusters)
				values = list(row.items())[0][1].split(',')
				len_values = len(values)

				if(nfeatures == None):
					nfeatures = [{} for x in range(len_values)]
					fcounter = np.ones(len_values, dtype=np.int32)

				int_values = np.zeros(len_values, dtype=np.int32)
				final_cluster = mclusters
				for i in range(len_values):
					if(i == 0):
						int_values[0] = int(values[0][1::])
					else:
						if(values[i] not in nfeatures[i]):
							features[values[i]] = fcounter[i]
							nfeatures[i][values[i]] = fcounter[i]
							fcounter[i] += 1
						# else:
						# 	nfeatures[i][values[i]] += 1

						int_values[i] = features[values[i]]
						if(int_values[i] not in final_cluster):
							final_cluster[int_values[i]] = {} if i < len_values-1 else []
						final_cluster = final_cluster[int_values[i]]

						# for j in range(i):
							
						# print(final_cluster)
						# if(int_values[i])

				final_cluster.append(mcounter)
				movies[mcounter] = int_values
				# print(int_values)
				mcounter += 1
				# if(mcounter == 10): break

			# print(movies)
			# print()
			# print(mclusters)
			# asdf

			columns = csv_reader._fieldnames[0].split(',')
			return movies, mclusters, columns, nfeatures, fcounter[1::]



	def input(self, dataset_name = "", skip_first_row = False):
		with open(self._dataset_path + dataset_name, newline='', encoding="utf-8") as csvfile:
			reader = csv.reader(csvfile, delimiter='\n', quotechar='|')
			result = []
			columns = []
			for ind, row in enumerate(reader):
				if(ind == 0 and skip_first_row):
					columns = row[0].split(",")
				else:
					result.append(row[0].split(","))
			# print(columns)
			return columns, result
		return [], []


	def output(self, dataset_name = "", rows = []):
		with open(self._dataset_path + dataset_name, 'w', newline='', encoding="utf-8") as csvfile:
			writer = csv.writer(csvfile, delimiter='\n', quotechar='|')
			writer.writerow(rows)


	@property
	def dataset_path(self):
		return self._dataset_path
	

