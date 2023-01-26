import csv
import sys
import random
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

class MovieGenerator:

	_io = None
	_data = None

	_all_genres = ['Science Fiction', 'Family', 'TV Movie', 'Adventure', 'Horror', 'Thriller', 
				  'Foreign', 'History', 'Fantasy', 'War', 'Western', 'Action', 'Crime', 'Animation',
				  'Comedy', 'Music', 'Mystery', 'Drama', 'Romance', 'Documentary']

	_top_companies = ['Lakeshore Entertainment', 'Monogram Pictures', 'Canal+', 'Home Box Office (HBO)', 'Walt Disney Productions', 'Nikkatsu', 'Daiei Studios', 'Working Title Films', 'Film i Väst', 'Dune Entertainment', 'RKO Radio Pictures', 'Westdeutscher Rundfunk (WDR)', 'Castle Rock Entertainment', 'Paramount Pictures', 'Channel Four Films', 'Warner Bros.', 'Miramax Films', 'Nordisk Film', 'Televisión Española (TVE)', 'Zentropa Entertainments', 'Film4', 'Columbia Pictures Corporation', 'Screen Gems', 'Universal International Pictures (UI)', 'Warner Bros. Animation', 'Millennium Films', 'Metro-Goldwyn-Mayer (MGM)', 'France 2 Cinéma', 'Orion Pictures', 'Columbia Pictures', 'HBO Films', 'Blumhouse Productions', 'United Artists', 'Pathé', 'Eurimages', 'Touchstone Pictures', 'Téléfilm Canada', 'New World Pictures', 'New Line Cinema', 'Fox Film Corporation', 'Nu Image Films', 'UK Film Council', 'Wild Bunch', 'Universal Pictures', 'British Broadcasting Corporation (BBC)', 'Shaw Brothers', 'Arte France Cinéma', 'DC Comics', 'Lionsgate', 'Svensk Filmindustri (SF)', 'The Rank Organisation', 'CinéCinéma', 'Twentieth Century Fox Film Corporation', 'Relativity Media', 'Amblin Entertainment', 'Arte', 'CJ Entertainment', 'Focus Features', 'BBC Films', 'Ciné+', 'StudioCanal', 'Gaumont', 'Walt Disney Pictures', 'TLA Releasing', 'Dimension Films', 'Summit Entertainment', 'Lions Gate Films', 'First National Pictures', 'Silver Pictures', 'Shôchiku Eiga', 'Morgan Creek Productions', 'Centre National de la Cinématographie (CNC)', 'BBC', 'The Weinstein Company', 'TriStar Pictures', 'Village Roadshow Pictures', 'Pixar Animation Studios', 'Zweites Deutsches Fernsehen (ZDF)', 'PolyGram Filmed Entertainment', 'Fine Line Features', 'DreamWorks SKG', 'Hallmark Entertainment', 'Lenfilm', 'New Regency Pictures', 'Hollywood Pictures', 'Regency Enterprises', 'France 3 Cinéma', 'Golden Harvest Company', 'Rai Cinema', 'TF1 Films Production', 'Toho Company', 'M6 Films', 'EuropaCorp', 'Fox 2000 Pictures', 'Fox Searchlight Pictures', 'Mosfilm', 'American International Pictures (AIP)', 'Imagine Entertainment', 'Canal+ España', 'Hammer Film Productions']

	_years = ["1900-1950", "1950-1960", "1960-1970", "1970-1980", "1980-1990", "1990-2000", "2000-2010", "2010-2017"]

	_votes = ['4-6', '6-8', '8-10']

	_langs = ['id', 'ko', 'jv', 'da', 'ur', 'bn', 'sh', 'bm', 'rw', 'el', 'no', 'et', 'pl', 'mr', 'hi', 'iu', 'sv', 'cy', 'mk', 'ab', 'sr', 'mn', 'si', 'ms', 'zu', 'eo', 'ta', 'nb', 'ay', 'tg', 'uz', 'he', 'qu', 'wo', 'cs', 'sk', 'la', 'fa', 'ne', 'hr', 'kk', 'lb', 'bo', 'sl', 'bg', 'hu', 'nl', 'ps', 'ro', 'bs', 'sq', 'xx', 'ka', 'th', 'fi', 'eu', 'lo', 'tl', 'sm', 'cn', 'ja', 'pa', 'ru', 'uk', 'ky', 'lv', 'vi', 'am', 'kn', 'te', 'de', 'lt', 'fy', 'hy', 'ar', 'ku', 'ca', 'is', 'pt', 'zh', 'af', 'mt', 'tr', 'ml', 'gl']

	_vcounts = ['>1000', '>100', '<100']



	def __init__(self, io = None, data = None):
		self._io = io
		self._data = data


	def _read_full_dataset(self, name="movies_metadata.csv", for_output = False):
		movies = []
		counter = {}
		with open(self._io.dataset_path + name, newline='', encoding='utf8') as csvfile:
			spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
			for ind, row in enumerate(spamreader):
				if(ind == 0):
					continue
				movie = list(csv.reader([row[0]], delimiter=","))[0]
				lgenres 	= movie[0]
				mid 		= movie[1]
				lang 		= movie[2]
				popularity 	= movie[3]
				lcompanies 	= movie[4]
				date 		= movie[5]
				title 		= movie[6]
				vote 		= movie[7]
				count 		= int(movie[8])

				genres = [self._all_genres.index(genre['name']) for genre in eval(lgenres)]
				companies = [comp['name'] for comp in eval(lcompanies)]
				year = 1980 if not date else int(date.split("-")[0])

				new_count = 0
				for x in [10, 100, 1000]:
					if(count < x):
						break
					new_count = x
				count = new_count

				if(for_output):
					# genres_str = "~".join([self._all_genres[i] for i in genres])
					# companies_str = "~".join([c for c in companies])
					genres_str = "" if len(genres)==0 else self._all_genres[genres[random.randint(0, len(genres)-1)]]
					companies_str = "" if len(companies)==0 else companies[random.randint(0, len(companies)-1)]
					new_movie = ["f"+str(ind), genres_str, companies_str, str(year), title, str(count)]
					new_movie = [x.replace(",",'') for x in new_movie]
					movies.append(','.join(new_movie))
				else:
					movies.append([genres, companies, year, title, count])
		columns = ["genre", "company", "year", "title", "count"]
		return columns, movies


	def _define_user(self):
		# ==================== GENRES ==================== #
		g_likes = set()
		g_dislikes = set()
		for i in range(len(self._all_genres)):
			rand = random.randint(0, 99)
			if(rand < 15): g_likes.add(self._all_genres[i])
			elif(rand < 20): g_dislikes.add(self._all_genres[i])

		# ==================== COMPANIES ==================== #
		c_likes = None
		c_dislikes = None
		if(random.randint(0,99) < 5):
			c_likes = self._top_companies[random.randint(0,len(self._top_companies)-1)]
		if(random.randint(0,99) < 2):
			c_dislikes = self._top_companies[random.randint(0,len(self._top_companies)-1)]

		# ==================== LANGUAGES ==================== #
		first_lang = True
		langs = set()
		new_lang = ''
		if False:
			for i in range(3):
				rand = random.randint(0,99)
				if(first_lang or rand < 10):
					while(new_lang in langs or first_lang):
						first_lang = False
						rand = random.randint(0,99)
						new_lang = 'en'
						if(rand < 60):
							if(rand < 10): new_lang = 'it'
							elif(rand < 20): new_lang = 'de'
							elif(rand < 30): new_lang = 'fr'
							elif(rand < 40): new_lang = 'es'
							else: 			 new_lang = self._langs[random.randint(0, len(self._langs) - 1)]
					langs.add(new_lang)

		# ==================== YEARS ==================== #
		# dislikes if not in range
		years_modes = ['2000+', '1990+', '1950+', '1900+']
		rand = random.randint(0,99)
		if(rand < 60): years_mode = 1
		elif(rand < 80): years_mode = 0
		elif(rand < 98): years_mode = 2
		else: years_mode = 3

		# ==================== POPULARITY ==================== #
		# dislikes if not in range
		pop_modes = ['1000+', '100+', '0+']
		if(rand < 80): pop_mode = 0
		elif(rand < 95): pop_mode = 1
		else: pop_mode = 2

		return [g_likes, g_dislikes, c_likes, c_dislikes, years_mode, pop_mode]


	def _simulate_user_vote(self, user, movie, exact_vote=False):

		# print(user)
		# print()
		# print(movie)
		# min_len = min(len(user[0]), len(movie["genre"]))
		genre_l_vote = 0
		if(len(user[0]) > 0):
			# genre_l_vote = int(len(set(user[0]) & set(movie["genres"])) * 100 / min_len)
			genre_l_vote = 100 if movie['genre'] in user[0] else 0

		# min_len = min(len(user[1]), len(movie["genres"]))
		genre_d_vote = 0
		if(len(user[1]) > 0):
			# genre_d_vote = int(len(set(user[1]) & set(movie["genres"])) * 100 / min_len)
			genre_d_vote = 100 if movie['genre'] in user[1] else 0

		company_l = 0
		if(user[2] and user[2] == movie["company"]):
			company_l = 100

		company_d = 0
		if(user[3] and user[3] == movie["company"]):
			company_d = 100

		years_modes = [2000, 1990, 1950, 1900]
		year = 0
		if(int(movie["year"]) >= years_modes[user[4]]):
			year = 100

		pop_modes = [1000, 100, 0]
		pop = 0
		# print(movie, user)
		if(int(movie["count"]) >= pop_modes[user[5]]):
			pop = 100

		votes = [genre_l_vote, 100 - genre_d_vote, year, pop]
		if(user[2]): votes.append(company_l)
		if(user[3]): votes.append(100 - company_d)
		
		vote = int(sum(votes) / len(votes))
		if(exact_vote):
			final_vote = max(0, min(100, vote))
		else:
			final_vote = max(0, min(100, vote + random.randint(-10, 10)))
		final_vote = min(final_vote+20, 100)
		return final_vote


	def _simulate_query_vote(self, user, query, exact_vote=False):
		# print(user, query)
		# aset = self._data.get_answer_set(query)
		aset = self._data.get_as_from_panda(query=query)
		# print(aset, end="\n\n")
		# asdf
		sums = 0
		final_vote = 0
		if(not aset.empty):
			for index, m in aset.iterrows():
				# vote = self._simulate_user_vote(user, self._data.films[m])
				vote = self._simulate_user_vote(user, m, exact_vote)
				sums += vote
			final_vote = int(sums / len(aset)) if len(aset) > 0 else 0
		# print(final_vote)
		return final_vote
		 

	def convert_movies(self, source="movies_metadata.csv", dest="films3.csv"):
		columns, movies = self._read_full_dataset(name=source, for_output = True)
		movies.insert(0, ",".join(columns))
		self._io.output(dest, movies)
		# print(movies)
		# asdf


	def generate_queries(self, size=1000, dest="queries.csv"):
		self._data.read_pd_inputs()

		df = self._data.movies_df
		df['count'] = pd.to_numeric(df["count"], errors='coerce')
		sorted_df = self._data.movies_df.sort_values(by=['count'], ascending = False)
		# print()
		# print(sorted_df.iloc[0])

		# g = [abs(int(random.gauss(-20, 10000))) for x in range(1000)]
		# print(min(g), max(g))

		# sys.exit(0)

		rows = []
		present_queries = {}
		for i in range(size):
			print(i)
			empty = True
			while empty:

				insert_title 	= False
				insert_genre 	= False
				insert_company 	= False
				insert_year 	= False
				insert_count 	= False


				insert_title = random.randint(0, 99) < 1
				insert_genre = random.randint(0, 99) < 60
				insert_company = random.randint(0, 99) < 20
				insert_year = random.randint(0, 99) < 60
				insert_count = random.randint(0, 99) < 20

				title, genre, company, year, count = None, None, None, None, None

				movie_id = None
				if(insert_title):
					movie_id = abs(int(random.gauss(-20, 10000)))
					if(movie_id >= df.shape[0]):
						insert_title = False
					else:
						title = df['title'][movie_id]
				if(insert_genre):
					if(insert_title): 
						if(df.iloc[movie_id]['genre']):
							insert_genre = False
						else:
							genre = df.iloc[movie_id]['genre']
					else: genre = self._all_genres[random.randint(0, len(self._all_genres) - 1)]
				if(insert_company):
					if(insert_title):
						if(df.iloc[movie_id]['company']):
							insert_company = False
						else:
							company = df.iloc[movie_id]['company']
					else: company = self._top_companies[random.randint(0, len(self._top_companies) - 1)]
				if(insert_year):
					if(insert_title): year = df.iloc[movie_id]['year']
					else:
						y = 0
						while y > 2017 or y < 1900:
							y = int(random.gauss(1990, 20))
						year = y
				if(insert_count):
					if(insert_title): count = df.iloc[movie_id]['count']
					else: count = [0, 10, 100, 1000][random.randint(0, 3)]

				empty = True not in [insert_title, insert_genre, insert_company, insert_year, insert_count]
				if(not empty):
					hash_set = hash(frozenset([genre, company, year, title, count]))
					if(hash_set in present_queries):
						empty = True
					else:
						row = []
						if insert_genre: row.append("genre="+genre)
						if insert_company: row.append("company="+company)
						if insert_year: row.append("year="+str(year))
						if insert_title: row.append("title="+str(title))
						if insert_count: row.append("count="+str(count))
						row.insert(0, "q"+str(i+1))

						aset = self._data.get_as_from_panda(query=row)
						if(aset.empty):
							empty = True
						else:
							present_queries[hash_set] = 1


			
			rows.append(",".join(row))
		self._io.output("queries.csv", rows)


	def generate_utilmat(self, size=10, dest="utilmat.csv"):
		# self._generate_queries()
		# asedf


		# _, movies = self._read_full_dataset()
		# movies.sort(key = lambda x: x[4], reverse=True)

		self._data.read_pd_inputs()


		# user has likes and dislikes
		#	3 - 5 likes 	-> 80-100
		#	0 - 2 dislikes  -> 0-40
		#	the remaining are mid vote

		# rated movies are mainly interesting ones, with some exceptions
		# user types:
		#	random watcher
		#	a lot of movies
		#	a lot of movies from same companies/popularity/year

		users = []
		starting = 0
		for i in range(size):

			# [g_likes, g_dislikes, c_likes, c_dislikes, years_mode, pop_mode]
			user = self._define_user()

			voted_queries = random.randint(0, 250)

			votes = {}
			for v in range(voted_queries):
				# movie_id = abs(int(random.gauss(0, 12000)))
				# while query_id in votes:
				# 	movie_id = abs(int(random.gauss(0, 12000)))

				# query_id = "q"+str(random.randint(1, len(self._data.queries)))
				query_id = random.randint(0, len(self._data.queries) - 1)
				# print(query_id)
				while query_id in votes:
					# query_id = "q"+str(random.randint(1, len(self._data.queries)))
					query_id = random.randint(0, len(self._data.queries) - 1)
				vote = self._simulate_query_vote(user, self._data.queries[query_id])
				votes[query_id] = str(vote)
				# print(vote)

			# arr = [abs(int(random.gauss(0, 12000))) for x in range(1000)]
			# print(self._data.queries)
			# print(votes)
			# print(self._data.queries)
			new_user = ["" if self._data.queries_ids[q[0]] not in votes else votes[self._data.queries_ids[q[0]]] for q in self._data.queries]
			new_user.insert(0, "u"+str(starting+i+1))
			# print(new_user)
			users.append(",".join(new_user))
			print(i)
		# print(users)


		self._io.output(dest, users)

		# print(users)
		sys.exit(0)
			




if __name__ == "__main__":
	generator = MovieGenerator()
	generator.generate()

	