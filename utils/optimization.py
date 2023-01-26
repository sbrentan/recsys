from annoy import AnnoyIndex
import random
import os
import csv
import pandas as pd
import numpy as np

class Optimization:

	def __init__(self, dir_path, test):
		self._dir_path = dir_path
		self._test = test
		pass

	def optimize_svd(self, dir_path):

		utilmat = pd.read_csv(dir_path + '/datasets/test_4/utilmat.csv')
		# scaler = StandardScaler()
		# standardized_data = scaler.fit_transform(df)
		utilmat = (utilmat.fillna(0) / 100).values
		# print(utilmat)
		# asdf


		from numpy import array
		from sklearn.decomposition import TruncatedSVD

		# svd
		svd = TruncatedSVD(n_components=500)
		svd.fit(utilmat)
		utilmat = svd.transform(utilmat)
		# print(utilmat)


		f = len(utilmat[0])  # Length of item vector that will be indexed

		t = AnnoyIndex(f, 'angular')
		for i in range(len(utilmat)):
		    # v = [random.gauss(0, 1) for z in range(f)]
		    t.add_item(i, utilmat[i])

		t.build(100) # 10 trees
		t.save('test.ann')

		# ...

		u = AnnoyIndex(f, 'angular')
		u.load('test.ann') # super fast, will just mmap the file
		# for i in range(1000):
		print(u.get_nns_by_item(0, 100, include_distances=True)) # will find the 1000 nearest neighbors


		# a = [0, 602, 963, 283, 6, 795, 764, 46, 536, 579, 974, 634, 871, 573, 951, 426, 451, 164, 178, 341, 56, 558, 948, 312, 953, 24, 571, 7, 958, 295, 132, 110, 17, 401, 535, 174, 729, 75, 74, 707, 928, 291, 72, 337, 155, 170, 471, 71, 797, 473, 776, 969, 989, 525, 325, 796, 734, 861, 90, 73, 123, 246, 418, 612, 239, 585, 968, 570, 188, 906, 42, 375, 640, 942, 610, 331, 857, 327, 376, 850, 407, 653, 55, 200, 873, 40, 952, 817, 926, 986, 872, 212, 144, 613, 742, 409, 387, 343, 27, 161]
		# b = [0, 409, 174, 602, 74, 759, 613, 338, 390, 69, 6, 123, 244, 861, 525, 984, 535, 486, 99, 963, 820, 800, 687, 682, 422, 56, 599, 24, 335, 951, 952, 315, 433, 196, 536, 796, 850, 707, 713, 961, 734, 983, 827, 990, 271, 377, 207, 926, 197, 40, 868, 573, 428, 327, 746, 880, 986, 204, 317, 564, 137, 840, 693, 132, 636, 577, 323, 353, 90, 642, 873, 896, 962, 379, 283, 193, 736, 46, 102, 225, 519, 401, 95, 714, 585, 39, 594, 818, 448, 71, 643, 725, 155, 128, 107, 164, 776, 458, 570, 492]
		# c = [0, 687, 486, 6, 979, 452, 781, 579, 174, 473, 963, 178, 930, 46, 246, 820, 69, 247, 353, 962, 613, 602, 7, 127, 767, 819, 585, 736, 123, 55, 643, 325, 796, 409, 281, 871, 944, 146, 204, 327, 724, 442, 132, 85, 75, 175, 498, 74, 774, 952, 906, 759, 762, 840, 24, 573, 237, 707, 39, 110, 201, 709, 889, 2, 56, 186, 861, 964, 561, 674, 721, 787, 426, 646, 941, 986, 38, 13, 522, 535, 49, 991, 942, 925, 838, 896, 660, 360, 881, 260, 837, 854, 337, 29, 766, 338, 527, 99, 418, 102]
		# d = [0, 409, 174, 602, 827, 74, 951, 613, 759, 687, 861, 6, 338, 244, 800, 585, 99, 123, 525, 796, 46, 390, 984, 535, 24, 196, 155, 350, 193, 448, 850, 56, 926, 327, 643, 599, 974, 707, 713, 983, 422, 328, 197, 682, 734, 875, 317, 90, 886, 986, 330, 579, 961, 955, 39, 71, 375, 377, 871, 539, 102, 975, 873, 132, 958, 746, 714, 889, 225, 642, 376, 335, 990, 40, 492, 379, 201, 323, 571, 594, 265, 797, 341, 128, 577, 742, 55, 764, 9, 271, 42, 653, 895, 677, 818, 107, 269, 412, 953, 337]
		a = [0, 1774, 28514, 9993, 7856, 6786, 26084, 16984, 5433, 3194, 2790, 27821, 22454, 29799, 24676, 29724, 14043, 24978, 24993, 4017, 12884, 9816, 7435, 9904, 27824, 17220, 5501, 15461, 2468, 21230, 14183, 28833, 27757, 3164, 23296, 29885, 24989, 9914, 28078, 11098, 17209, 180, 16070, 12715, 11550, 25697, 18947, 24394, 3318, 6702, 28918, 16682, 12865, 2316, 655, 17781, 8878, 26840, 22355, 13080, 5353, 27543, 1266, 7046, 10452, 24544, 26952, 8241, 25914, 1497, 17301, 11284, 6559, 19277, 9755, 17278, 20806, 6749, 23019, 22558, 19153, 335, 4251, 14175, 28903, 1742, 19521, 15608, 24637, 1879, 22668, 27980, 15330, 3744, 8080, 21346, 12368, 9208, 28845, 12056]
		a = [0, 17937, 7576, 20499, 1774, 27821, 26152, 12851, 4089, 25598, 22454, 11756, 16664, 1809, 28514, 24310, 9829, 24436, 15962, 10250, 11591, 23143, 18027, 5433, 3545, 20295, 21748, 19053, 19517, 28554, 13955, 387, 21230, 19404, 20330, 24808, 11276, 13064, 17397, 12012, 24967, 27562, 5741, 17291, 7856, 29799, 12828, 20847, 9088, 18997, 6743, 16605, 27034, 22964, 2582, 9299, 16984, 730, 25061, 17702, 23239, 5518, 20412, 13731, 3944, 22082, 20156, 13941, 24210, 251, 13202, 20535, 19419, 18479, 4243, 8740, 12117, 24595, 12884, 13458, 4269, 461, 16894, 14536, 7342, 12865, 9054, 9816, 10082, 20364, 21729, 4017, 22505, 24285, 16070, 14657, 7622, 7845, 27005, 10697]
		b = [0, 776, 24210, 22505, 13389, 13080, 23210, 11591, 10667, 20847, 657, 16605, 6521, 23299, 28270, 19194, 9296, 15836, 25273, 1266, 5856, 650, 11417, 10355, 11331, 18912, 21175, 20007, 15050, 20226, 17990, 25743, 25343, 13551, 16682, 9644, 1703, 24108, 9636, 12611, 23827, 10542, 27813, 1440, 12102, 9042, 5224, 25117, 3698, 15763, 16307, 28791, 15819, 6949, 7891, 13290, 6167, 14419, 14175, 6603, 24688, 3483, 13124, 27158, 5676, 2696, 6907, 832, 27554, 3727, 5062, 23443, 25784, 24924, 18127, 3519, 15366, 1762, 29947, 29386, 15414, 26433, 27623, 26540, 17661, 26470, 22166, 17426, 2025, 26534, 5983, 20137, 18143, 21863, 26213, 16428, 28261, 9023, 1305, 1054]
		print()
		print(len([value for value in b if value in a]) / len(set(b + a)))

		s1 = 0
		s2 = 0
		for i in range(100):
			s1 += u.get_distance(0, a[i])
			s2 += u.get_distance(0, b[i])
		s1 /= 100
		s2 /= 100
		print(s1, s2)

	def optimize_clustering(self, dir_path):

		utilmat = pd.read_csv(dir_path + '/datasets/test_4/utilmat.csv')
		utilmat = (utilmat.fillna(0) / 100).values
		utilmat = utilmat.round(0).astype(np.int8)
		print(utilmat)
		

		# from sklearn.cluster import AgglomerativeClustering
		# clustering = AgglomerativeClustering().fit(utilmat)
		# print(clustering.labels_)


		s = utilmat.sum(axis=0)
		print(len(s))
		print(s)
		# asdf


		X = np.array([[1, 1, 1],
					  [1, 0, 1],
					  [1, 1, 0],
					  [0, 1, 1],
					  [1, 0, 0],
					  [0, 0, 1],
					  [0, 1, 0]])
		# X = utilmat
		S = X.sum(axis=1)
		print(S)
		C = np.matmul(X,X.T)
		result = (C / S).round(2)
		print(result)


		asdf
		utilmat = utilmat.round(0).astype(np.int8)
		for i in range(10):
			u = random.randint(0, len(utilmat) - 1)
			nz = utilmat[u].nonzero()

	def optimize_mat(self, mat, ncertificates, nsplits):
		# utilmat = pd.read_csv(self._dir_path + '/datasets/'+self._test+'/utilmat.csv')
		utilmat = (mat / 100).round(0).astype(np.int8)

		# ncertificates = 10
		lclusters = [{} for i in range(ncertificates)]
		user_clusters = { u : [] for u in utilmat.index}
		for c in range(ncertificates):
			# nsplits = 8
			splits = []
			for splitid in range(nsplits):
				split1, split2 = [], []
				rand = random.randint(0, len(utilmat.iloc[0]) - 1)
				splits.append(rand)

			combs = {}
			clusters = {}
			stack = [[] for i in range(nsplits)]
			combinations = pow(2, nsplits)
			for i in range(combinations):
				strbits = bin(combinations+i)[3::]
				bits = list(map(lambda x: int(x), list(strbits)))
				new = utilmat


				curr_comb = ''
				for ind, b in enumerate(bits):
					curr_comb += str(b)
					if(curr_comb in combs):
						new = combs[curr_comb]
					else:
						new = new[new.iloc[:, splits[ind]] == b]
						combs[curr_comb] = new
				for u in new.index:
					user_clusters[u].append(strbits)
				clusters[strbits] = list(new.index)

			lclusters[c] = clusters
			# print(len(clusters['0'*nsplits]))
			# print(list({k : len(v)} for k, v in clusters.items()))
		return lclusters

		# empty_certificates = []
		# for el in lclusters[0]['0'*nsplits]:
		# 	empty = True
		# 	for lclust in lclusters[1::]:
		# 		if(el in lclust['0'*nsplits]):
		# 			empty = False
		# 			break
		# 	if(empty): empty_certificates.append(el)
		# print()
		# print(len(empty_certificates), empty_certificates)

		# print(list({ u : sum([len(i) for ind, i in enumerate(lclusters) if uv[ind] != '0'*nsplits]) } for u, uv in user_clusters.items()))

