import os
import sys
import itertools
import numpy as np
import pandas as pd
import time
import dask.dataframe

from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

from utils.io_manager import IOManager
from utils.generator import MovieGenerator
from utils.data_manager import DataManager
from utils.cosine_similarity import CosineSimilarity

class Recommendator:

    _io = None
    _data = None

    CONVERT_MOVIES   = False
    GENERATE_QUERIES = False
    GENERATE_VOTES  = False

    SKIP_READINGS = True

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._io = IOManager(dir_path + "/datasets/")
        self._data = DataManager(self._io)

        t = time.time()
        if(self.CONVERT_MOVIES or self.GENERATE_QUERIES or self.GENERATE_VOTES):
            generator = MovieGenerator(self._io, self._data)
            if(self.CONVERT_MOVIES):   generator.convert_movies(source="movies_metadata.csv", dest="films.csv")
            if(self.GENERATE_QUERIES): generator.generate_queries(dest="queries.csv")
            if(not self.SKIP_READINGS): self._data.read_inputs()
            if(self.GENERATE_VOTES):  generator.generate_utilmat(size=1, dest="utilmat5.csv")
        else:
            if(not self.SKIP_READINGS): self._data.read_inputs()

        # self._data.read_utilmat()
        print("Time to read: " + str(round(time.time() - t, 2)))
        # sys.exit(0)

    # Computes feedback values for a single user, using affinity matrix between queries
    def _using_affinity_matrix(self):
        print("\n===== AFFINITY MATRIX =====\n")
        for_user = list(self._data.utilmat['users'].items())[0][0]
        aff_mat = {}
        for qid, q in self._data.queries.items():
            items = self._data.get_answer_set(qid)
            q["as_ids"] = items
            for fid in items:
                i = self._data.films[fid]
                if("queries" in i): i["queries"].add(qid)
                else: i["queries"] = set([qid])
                if('score' not in i): i['score'] = 0
                user_fb = int(self._data.utilmat['users'][for_user][self._data.utilmat['queries'][qid]])
                i['score'] = max(user_fb, i['score'])
        for fid, i in self._data.films.items():
            comb = itertools.combinations(i["queries"],2)
            for c in comb:
                first, second = (c[1], c[0]) if c[0] > c[1] else (c[0], c[1])
                if(first not in aff_mat): aff_mat[first] = {}
                if(second not in aff_mat[first]):
                    aff_mat[first][second] = 1
                else:
                    aff_mat[first][second] += 1
        for uid, fblist in self._data.utilmat['users'].items():
            for qid, qindex in self._data.utilmat['queries'].items():
                for qid, q in self._data.queries.items():
                    if(q['as_ids']):
                        q['score'] = float(sum(list(map(lambda x: self._data.films[x]['score'], q['as_ids'])))) / len(q['as_ids'])
                    else:
                        q['score'] = 0
        print([(qid, q['score']) for qid, q in self._data.queries.items()])


    # Computes feedback values for every user, using correspondances between queries feedbacks
    def _using_correspondance_matrix(self):
        print("\n===== CORRESPONDANCE MATRIX =====\n")

        queries_len = len(self._data.utilmat['queries'])
        corr_mat = np.zeros( (queries_len, queries_len))
        queries_count = np.zeros( (queries_len, queries_len))
        for uid, flist in self._data.utilmat['users'].items():
            comb = itertools.combinations([x for x in list(enumerate(flist)) if x[1] >= 0],2)
            asdf
            print(list(comb))
            for c in comb:
                id1, fb1 = c[0] if c[0][0] < c[1][0] else c[1] # 0 80
                id2, fb2 = c[1] if c[0][0] < c[1][0] else c[0] # 2 60
                print(fb2, fb1)
                print(corr_mat)
                print(id1, id2)
                result = float((1 - fb2/fb1) + (corr_mat[id1][id2] * queries_count[id1][id2])) / (queries_count[id1][id2] + 1)
                corr_mat[id1][id2] = round(result, 3)
                print(id1, id2, '->', corr_mat[id1][id2])
                queries_count[id1][id2] += 1
                # 80-60 = 20 -> 0 * 0 + (-0.25) / 1  -> -0.25
                # -> -0.25
                # 80-40 = 40 -> (-0.25) * 80 * 1 + 40 / 2 -> 30
                # (-0.25) * 1 + (-0.5) / 2 -> -0.37 * 80
                # 80-40 = 40 -> 20 * 1 + 40 / 2 -> 30
                # 80-60 = 20 -> 30 * 2 + 20 / 3 -> 26.6

                # 80 + 60 / 2 -> 70
                # 80, 40 / 2 -> 60 + 70 * 2 / 3 -> 6
        print(corr_mat)
        for uid, flist in self._data.utilmat['users'].items():
            enumerated = list(enumerate(flist))
            evaluated_queries = [x for x in enumerated if x[1] >= 0]
            empty_queries = [x for x in enumerated if x[1] < 0]
            for empty_fb in empty_queries:
                avg_fb = 0
                avg_count = 0
                for fb in evaluated_queries:
                    first, second = (fb[0], empty_fb[0]) if fb[0] < empty_fb[0] else (empty_fb[0], fb[0])
                    print(first, second, corr_mat[first][second], fb[1])
                    if(corr_mat[first][second] == 0):
                        continue
                    avg_fb += fb[1] - corr_mat[first][second]*fb[1]
                    avg_count += 1
                    if(avg_count > 0):
                        flist[empty_fb[0]] = int(avg_fb / avg_count)
        print(self._data.utilmat)


    # Computes top k similar users and get average votes for every user
    def _using_users_similarity(self):
        # t = time.time()
        # cols, mat = self._io.input_csr_matrix('utilmat2.csv')
        # print("Time to read csr matrix: "+str(round(time.time() - t, 2)))
        mat = self._data.

        cossim = CosineSimilarity()
        clusters, empty_clusters = cossim.compute(mat)

        mat = mat.todense()

        t = time.time()
        for c, values in clusters.items():
            # calculate average votes for the clustered users
            a = np.empty((len(clusters[0]), mat.shape[1]))
            counter = np.zeros((1, mat.shape[1]))
            for ind, userid in enumerate(values):
                b = mat[userid-1]
                counter += b.astype(bool)
                a[ind] = b

            summed_values = a.sum(axis=0)
            result  = summed_values / counter
            # sys.exit(0)
            # counter = [len(a[:, x].nonzero()[0]) for x in range(mat.shape[1])]

            # a = lil_matrix((len(clusters[0]), mat.shape[1]), dtype=np.int8)
            # np.set_printoptions(threshold=sys.maxsize)
            # col_values = a[:, 0]
            # print(counter)
            # asdf
            # result = round(sum(col_values) / len(col_values.nonzero()[0]), 2)
        print("Time to compute utilmat values: " + str(round(time.time() - t, 2)))


    def _as_test(self):
        # print(self._data.queries)
        print(self._data.movies[7106])
        # print(self._data.mclusters[93][981][4][1540][377])
        query_id = self._data.queries_ids["q20"]
        t = time.time()
        # print(query_id, self._data.queries[query_id])
        # print(self._data.get_answer_set(query_id))
        for q in self._data.queries:
            self._data.get_answer_set(query=q)
        print('Time:', str(round(time.time() - t, 3)))
        
        # self._data.read_inputs_old()
        t = time.time()
        for q in self._data.queries:
            self._data.get_as_from_movies(query=q)
        print('Time:', str(round(time.time() - t, 3)))

    def _compute_queries_clusters(self):
        movies_indexes = self._data.movies_df.index
        self._data.movies_df.index = [x for x in range(len(movies_indexes))]

        # print(self._data.movies_df)
        # asdf

        cols = self._data.movies_df.columns
        shingles = []
        c = 0
        for q in self._data.queries:
            # panda = self._data.movies_df
            # for cond in q[1::]:
            #     col, val = cond.split("=")
            #     panda = panda[panda[col] == val]
            #     # l.append(col+" == '"+val+"'")
            panda = self._data.get_as_from_panda(query=q)
            new = np.zeros(self._data.movies_df.shape[0])
            if(not panda.empty):
                indexed = panda.index
                # print(self._data.queries_idsself._queries[i][0])
                for e in indexed:
                    # print(panda.index)
                    # new[self._data.movies_ids[e]] = 1
                    new[e] = 1
            else:
                c += 1
            shingles.append(new)

        # print(shingles[0].sum())
        # asdf
        print(c)
        # print(len(shingles))
        cossim = CosineSimilarity()
        clusters, empty = cossim.compute(csr_matrix(shingles, dtype=np.int32))

        return clusters, empty
        # time.sleep(10)


    def _compute_query_value(qid, feedback):
        # 537: [320, 81, 40, 75, 257]
        q = self._data.queries[qid]

        # user_avg_score = get average feedback for interested user
        # for every query feedbacked by user:
        #       --- find most relevant features
        #       for every feature:
        #               fvalues[feature] += (query_score - user_avg_score)
        #               fcount[feature] += 1
        #       --- add frequency adjustement
        #       for f in fvalues:
        #               fvalues[f] = int(fvalues[f] / fcount[f])
        #               fvalues[f] = fvalues[f] - (int(len(fvalues) - fcount[f] / len(fvalues)) * fvalues[f])
        #       --- now fvalues[f] contains avg votes(up to 100) for every feature
        #       --- one fvalues list has to be created for every user
        # --- Iterate candidate queries and evaluate them according to precomputed fvalues
        # for every qid in query candidates:
        #       query_avg_score = get average feedback among all users of qid
        #       score_query, count_query = 0, 0
        #       for every movie in aset(qid):
        #           score_movie, count_movie = 0, 0 
        #           for feature in movie:
        #               score_movie += fvalues[feature]
        #               count_movie += 1
        #           score_movie /= count
        #           score_query += score_movie
        #           count_query += 1
        #       score_query /= count_query
        # --- Now we have scored queries from content-based filtering


        # --- Through cossim user similarity, compute top k similar users
        # --- Compute mean of feedbacked queries(standardized) of those users
        # --- Now we have scored queries from collaborative filtering


        # --- Get final result by averaging these votes through a weighted scheme


        # --- For remaining queries: get avg score given by users


        # --- AT RUNTIME(for user x get top k queries):
        # --- Filter out user-feedbacked queries( = 0)
        # --- Order row x of precomputed utility matrix descending
        # --- Get top k queries



        # --- COLD START REMOVED THANKS TO USAGE OF 'POPULARITY'





    def _using_mat_avgs(self):
        self._data.read_pd_inputs()
        df = self._data.utilmat_df

        # print(df)

        q_means = df.mean(axis = 0, skipna = True)
        u_means = df.mean(axis = 1, skipna = True)

        print(q_means)
        print('-----')
        print(u_means)

        q = self._data.queries[100]
        # print(q)
        result = self._data.get_as_from_panda(query=q)
        # print(result)

        # bool_df = df.notna()
        # small_df = df.iloc[:5,:5]
        # print(small_df)
        # print(small_df.mean(axis = 0, skipna = True))


    def _using_lsh(self):
        self._data.read_pd_inputs()
        from utils.lsh_similarity import LSHSimilarity
        asets = []
        for i in range(len(self._data.queries)):
            aset = self._data.get_as_from_panda(query=self._data.queries[i])
            if(not aset.empty):
                # if(i == 999):s
                    # print(list(aset.index))
                print("++++++")
                asets.append(list(aset.index))
            if(i == 50):
                break
                # asets.append(aset.index)
                # print(aset.index)
                # for s in aset.index:
                #     print(s)
        # print(asets)
        # asdf

        num_permutations = 128
        forest = LSHSimilarity.get_forest(asets, num_permutations)

        num_recommendations = 5
        # aset = ['f537','f755','f895','f1050','f1065','f1066','f1113','f1245','f1249','f1253','f1280','f1299','f1313','f1326','f1333','f1516','f1868','f1876','f1886','f1960','f1996','f1997','f2002','f2031','f2032','f2145','f2146','f2176','f2291','f2298','f2406','f2545','f2554','f2643','f2678','f2804','f2840','f2855','f2901','f2931','f2952','f2969','f3093','f3104','f3196','f3312','f3422','f3471','f3814','f3915','f3917','f4153','f4234','f4291','f4408','f4569','f4729','f4765','f4786','f4800','f4857','f4909','f4920','f5155','f5433','f5522','f5574','f5580','f5595','f5622','f5629','f5641','f5761','f5781','f5782','f5783','f5784','f5785','f5786','f5787','f5788','f5789','f5790','f5791','f5792','f5793','f5794','f5795','f5796','f5797','f5798','f5799','f5801','f5802','f5803','f5864','f5900','f5937','f5938','f5940','f5941','f5942','f5943','f5944','f5945','f5946','f5947','f5948','f5949','f5950','f5951','f5952','f5953','f5954','f5955','f5956','f5957','f5958','f5959','f5960','f5961','f5962','f5963','f5964','f5965','f5966','f5967','f5968','f5969','f5970','f5971','f5972','f5973','f5974','f5975','f5976','f5978','f5979','f5982','f5983','f5984','f5985','f5987','f5988','f5989','f5990','f5991','f5992','f5993','f5994','f5995','f5996','f5997','f5998','f6001','f6002','f6003','f6004','f6005','f6006','f6007','f6009','f6010','f6011','f6012','f6013','f6014','f6016','f7204','f8024','f8808','f8811','f8812','f9585','f9781','f9941','f10752','f11234','f11466','f11581','f12392','f12528','f12673','f12677','f13668','f14114','f14455','f14601','f14605','f14792','f15119','f15513','f15762','f15946','f16083','f16372','f16498','f16669','f16919','f17063','f17107','f18175','f18203','f18220','f18241','f18495','f18769','f19131','f19300','f19598','f19728','f19787','f19819','f19911','f19951','f19957','f20857','f20867','f21215','f21360','f21703','f21931','f22137','f23124','f23296','f23372','f24261','f24288','f24384','f24450','f24754','f25313','f25364','f25372','f25515','f25516','f25674','f25675','f25903','f26035','f26264','f26341','f26460','f26895','f27008','f27021','f27276','f27304','f27358','f27491','f27530','f27893','f27960','f28100','f28253','f28264','f28293','f28520','f28557','f29172','f29174','f29188','f29260','f29286','f29471','f29549','f29611','f29833','f30066','f30141','f30163','f30177','f30328','f30329','f30409','f30484','f30555','f30659','f30827','f30828','f30868','f31092','f31395','f31600','f31708','f31799','f31800','f32080','f32090','f32189','f32522','f32610','f32769','f32786','f32881','f32982','f33220','f33334','f33616','f34006','f34233','f34255','f34263','f34681','f34704','f34708','f34735','f34741','f34941','f35008','f35188','f35271','f35302','f35315','f35729','f36433','f36541','f36614','f36657','f36711','f36889','f37102','f37106','f37325','f37828','f37964','f38043','f38087','f38798','f38998','f39026','f39048','f39103','f39343','f39456','f39480','f39709','f39761','f39835','f39856','f39968','f40155','f40275','f40859','f40917','f40984','f41648','f41667','f41834','f41913','f42069','f42667','f42847','f42990','f42991','f43210','f43374','f43520','f44106','f44173','f44419','f44499','f44520','f44567','f44569','f45015','f45069','f45158']
        # aset = ['f8109', 'f31503']
        aset = ['f537', 'f755', 'f895', 'f1050', 'f1065', 'f1066', 'f1113', 'f1245', 'f1249', 'f1253', 'f1280', 'f1299', 'f1313', 'f1326', 'f1333', 'f1516', 'f1868', 'f1876', 'f1886', 'f1960', 'f1996', 'f1997', 'f2002', 'f2031', 'f2032', 'f2145', 'f2146', 'f2176', 'f2291', 'f2298', 'f2406', 'f2545', 'f2554', 'f2643', 'f2678', 'f2804', 'f2840', 'f2855', 'f2901', 'f2931', 'f2952', 'f2969', 'f3093', 'f3104', 'f3196', 'f3312', 'f3422', 'f3471', 'f3814', 'f3915', 'f3917', 'f4153', 'f4234', 'f4291', 'f4408', 'f4569', 'f4729', 'f4765', 'f4786', 'f4800', 'f4857', 'f4909', 'f4920', 'f5155', 'f5433', 'f5522', 'f5574', 'f5580', 'f5595', 'f5622', 'f5629', 'f5641', 'f5761', 'f5781', 'f5782', 'f5783', 'f5784', 'f5785', 'f5786', 'f5787', 'f5788', 'f5789', 'f5790', 'f5791', 'f5792', 'f5793', 'f5794', 'f5795', 'f5796', 'f5797', 'f5798', 'f5799', 'f5801', 'f5802', 'f5803', 'f5864', 'f5900', 'f5937', 'f5938', 'f5940', 'f5941', 'f5942', 'f5943', 'f5944', 'f5945', 'f5946', 'f5947', 'f5948', 'f5949', 'f5950', 'f5951', 'f5952', 'f5953', 'f5954', 'f5955', 'f5956', 'f5957', 'f5958', 'f5959', 'f5960', 'f5961', 'f5962', 'f5963', 'f5964', 'f5965', 'f5966', 'f5967', 'f5968', 'f5969', 'f5970', 'f5971', 'f5972', 'f5973', 'f5974', 'f5975', 'f5976', 'f5978', 'f5979', 'f5982', 'f5983', 'f5984', 'f5985', 'f5987', 'f5988', 'f5989', 'f5990', 'f5991', 'f5992', 'f5993', 'f5994', 'f5995', 'f5996', 'f5997', 'f5998', 'f6001', 'f6002', 'f6003', 'f6004', 'f6005', 'f6006', 'f6007', 'f6009', 'f6010', 'f6011', 'f6012', 'f6013', 'f6014', 'f6016', 'f7204', 'f8024', 'f8808', 'f8811', 'f8812', 'f9585', 'f9781', 'f9941', 'f10752', 'f11234', 'f11466', 'f11581', 'f12392', 'f12528', 'f12673', 'f12677', 'f13668', 'f14114', 'f14455', 'f14601', 'f14605', 'f14792', 'f15119', 'f15513', 'f15762', 'f15946', 'f16083', 'f16372', 'f16498', 'f16669', 'f16919', 'f17063', 'f17107', 'f18175', 'f18203', 'f18220', 'f18241', 'f18495', 'f18769', 'f19131', 'f19300', 'f19598', 'f19728', 'f19787', 'f19819', 'f19911', 'f19951', 'f19957', 'f20857', 'f20867', 'f21215', 'f21360', 'f21703', 'f21931', 'f22137', 'f23124', 'f23296', 'f23372', 'f24261', 'f24288', 'f24384', 'f24450', 'f24754', 'f25313', 'f25364', 'f25372', 'f25515', 'f25516', 'f25674', 'f25675', 'f25903', 'f26035', 'f26264', 'f26341', 'f26460', 'f26895', 'f27008', 'f27021', 'f27276', 'f27304', 'f27358', 'f27491', 'f27530', 'f27893', 'f27960', 'f28100', 'f28253', 'f28264', 'f28293', 'f28520', 'f28557', 'f29172', 'f29174', 'f29188', 'f29260', 'f29286', 'f29471', 'f29549', 'f29611', 'f29833', 'f30066', 'f30141', 'f30163', 'f30177', 'f30328', 'f30329', 'f30409', 'f30484', 'f30555', 'f30659', 'f30827', 'f30828', 'f30868', 'f31092', 'f31395', 'f31600', 'f31708', 'f31799', 'f31800', 'f32080', 'f32090', 'f32189', 'f32522', 'f32610', 'f32769', 'f32786', 'f32881', 'f32982', 'f33220', 'f33334', 'f33616', 'f34006', 'f34233', 'f34255', 'f34263', 'f34681', 'f34704', 'f34708', 'f34735', 'f34741', 'f34941', 'f35008', 'f35188', 'f35271', 'f35302', 'f35315', 'f35729', 'f36433', 'f36541', 'f36614', 'f36657', 'f36711', 'f36889', 'f37102', 'f37106', 'f37325', 'f37828', 'f37964', 'f38043', 'f38087', 'f38798', 'f38998', 'f39026', 'f39048', 'f39103', 'f39343', 'f39456', 'f39480', 'f39709', 'f39761', 'f39835', 'f39856', 'f39968', 'f40155', 'f40275', 'f40859', 'f40917', 'f40984', 'f41648', 'f41667', 'f41834', 'f41913', 'f42069', 'f42667', 'f42847', 'f42990', 'f42991', 'f43210', 'f43374', 'f43520', 'f44106', 'f44173', 'f44419', 'f44499', 'f44520', 'f44567', 'f44569', 'f45015', 'f45069', 'f45158']
        result = LSHSimilarity.predict(aset, None, num_permutations, num_recommendations, forest)
        print('\nTop Recommendation(s) is(are) \n', result)
        sys.exit(0)



    def _hybrid(self):
        self._data.read_pd_inputs()
        q_means = df.mean(axis = 0, skipna = True)
        u_means = df.mean(axis = 1, skipna = True)

        queries_clusters = self._compute_queries_clusters()




    def recommend(self):

        # self._using_users_similarity()

        self._using_query_similarity()

        # self._using_mat_avgs()

        # self._using_lsh()

        self._hybrid()


        # CONTENT BASED
        # create query-query affinity matrix
        # for every query:
        #       assign to the reachable items the query id
        # for every item:
        #       for every pair of reachable queries: add +1 in the affinity matrix
        # compute transitive affinity alongside user feedback
        # for every feedbacked query:
        #       get most affine query
        #       if feedback not present in utilmat:
        #          assign score to that query in utilmat by a factor of (score/100)
        #           --->>>> PROBLEM: does not account for single film score
        #                --->>> SOLUTION: change query score to average from film score (assigned from user feedback) 
        # self._using_affinity_matrix()



        # HYBRID BASED
        # cluster similar users together
        #       1) TFIDF based on film feedbacks
        #       2) Cluster film together and observe users feedbacks on those clusters
        #           * Films can be grouped together based on similar attributes? and names?
        # compute content based on representative user for every cluster



        # QUERY CORRESPONDANCE
        # find queries correspondance from each other
        # create query-query correspondance matrix:
        # for every user:
        #       for every feedback:
        #               find other feedback
        #               compute correspondance similarity between said queries
        # for every user:
        #       for every evaluated query:
        #               for every other query:
        #                       compute evaluation based on qq_correspondance matrix
        #               find average of those evaluations
        # self._using_correspondance_matrix()





        # save similarity of pair of items inside sparse matrix
        # for every answer set:
        #       raise/decrease similarity of each pair of items by an amount related to feedback (e.g. + (feedback-50) / 50)
        # 
        # compute matrix from item similars and user feedback
        # for every user answer set:
        #       give score to most connected items up to a maximum of user feedback
        # fetch items with highest score
        # compute query that better represents those items

        print()
        pass

        

if __name__ == "__main__":
    t = time.time()
    recommendator = Recommendator()
    recommendator.recommend()
    print(time.time() - t)


# WHAT HAPPENS WHEN N. QUERIES < N. UTILMAT