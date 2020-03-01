# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import re
import numpy as np
import nltk
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize
from deps import PorterStemmer
import collections

# noinspection PyMethodMayBeStatic


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'Chadbot'
        # Characteristics
        # Bro, Red Solo Cups, Beer Pong, Hat backwards, Fraternity, Confident in his body, Thirsty, Sees women as objects,
        # you get it dawg
        # netflix and chill bro
        # fake instahandle
        #
        self.p = PorterStemmer.PorterStemmer()
        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        # dictionary of {word:sentiment}

        sent_list = movielens.sentiment()
        stemmed_list = {}
        for w, s in sent_list.items():
            stemmed = self.p.stem(w)
            if s == "neg":
                stemmed_list[stemmed] = -1
            elif s == "pos":
                stemmed_list[stemmed] = 1

        self.sentiment = stemmed_list
        #############################################################################
        # TODO: Binarize the movie ratings matrix.                                  #
        #############################################################################
        # ratings = [["title", "genre"]]
        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = ratings
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        # Request for information
        # Confirm Information Understanding
        # Request for Clarification
        # Offer Recommendation
        # Respond to random statements
        #

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = "How can I help you?"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = "Have a nice day!"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message

    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
        else:
            response = "I processed {} in starter mode!!".format(line)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information from a line of text.

        Given an input line of text, this method should do any general pre-processing and return the
        pre-processed string. The outputs of this method will be used as inputs (instead of the original
        raw text) for the extract_titles, extract_sentiment, and extract_sentiment_for_movies methods.

        Note that this method is intentially made static, as you shouldn't need to use any
        attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        #############################################################################
        # TODO: Preprocess the text into a desired format.                          #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to your    #
        # implementation to do any generic preprocessing, feel free to leave this   #
        # method unmodified.                                                        #
        #############################################################################

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(
              chatbot.preprocess('I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        titles = []

        if self.creative:

            for i in range(len(self.titles)):

                movie_title = self.titles[i][0]

                # get everything before (
                movie_title_edited = re.findall(
                    '(.*) \([0-9]{4}\)', movie_title)

                if len(movie_title_edited) != 0:
                    movie_title_edited = movie_title_edited[0]

                else:
                    movie_title_edited = movie_title

                movie_title_lc = movie_title_edited.lower()

                preprocessed_input_lc = preprocessed_input.lower()

                movie_title_use = movie_title_lc

                articles = ['the', 'an', 'a']

                for article in articles:

                    pattern = ', ' + article + '(?:\W|$)'

                    if re.findall(pattern, movie_title_lc) != []:

                        suffix = ', ' + article

                        movie_title_use = re.sub(
                            suffix, "", movie_title_use)  # space??
                        movie_title_use = article + ' ' + movie_title_use

                        break

                # word_tokens = word_tokenize(s) THEN CUT WORDS FROM EACH END

                pattern = '(?:[\W|^])' + \
                    re.escape(movie_title_use) + '(?:[\W|$])'

                if re.findall(pattern, preprocessed_input_lc) != []:
                    titles.append(movie_title_use)

                # GO FROM EITHER SIDE OF MOVIE, TAKE OFF PIECE AT A TIME
                # WOULDN'T NEED EXTRA ARTICLE PROCESSING

        else:
            # FOR SENTIMENT!!
            titles = re.findall('"([^"]*)"', preprocessed_input)

        return titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.

        Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        title = title.lower()

        name = title
        year = None
        # separate title and year if they give one
        if (title.find('(') != -1) and (title.find(')') != -1):
            name = title.split('(')[0].strip()
            year = title.split('(')[1][:-1]

        # remove articles from input title
        # skipping the foreign articles that clash with english words until later
        articles = []
        if self.creative:
            articles = ["the ", "an ", "a ", "la ", "le ", "les ", "las ", "los ",
                        "l'", "il ", "der ", "el ", "lo ", "das ", "det ", "en ", "une ", "den "]  # , "die ", "i "]
        else:
            articles = ["the ", "an ", "a "]

        for article in articles:
            if name[0:len(article)] == article:
                name = name[len(article):]
                break

        total_matches = {}

        for i in range(len(self.titles)):
            cur_title = self.titles[i][0].lower()
            cur_year = None
            # find the thing that matches ([0123456789]{4})
            # matches = re.findall("\(([0123456789]{4})\)", cur_title)
            # print(year_start.group(), cur_title)
            pat = re.compile("\([0123456789]{4}\)")
            for match in pat.finditer(cur_title):
                cur_year = match.group()[1:-1].strip()

            match = []
            # regex substring match
            if self.creative:
                # (space/comma/right_paren right afterwards to cut it off)
                creative_pat = name + "[ ,):].*\([0123456789]{4}\)"
                match = re.findall(creative_pat, cur_title)

                die_pat = "\(" + name[len("die "):] + \
                    ", die\) \([0123456789]{4}\)"
                match_die = re.findall(die_pat, cur_title)
                if (len(match_die) > 0):
                    for item in match_die:
                        match.append(item)

                i_pat = name[len("i "):] + ", i\) \([0123456789]{4}\)"
                match_i = re.findall(i_pat, cur_title)
                if (len(match_i) > 0):
                    for item in i_pat:
                        match.append(item)
            else:
                # to exactly the thing it gives us, articles, (foreign title), a space, then parenthesis for the year
                starter_pat = "^(a |the )?" + name + \
                    "(, (the|an|a))?( \(.*\))+"
                match = re.findall(starter_pat, cur_title)

            if year != None and len(match) > 0:
                # they do give a year, check the matched title from the previous line to see if it contains the year. this ordering prevents having to reconstruct with articles, etc.)
                match = re.findall(year, cur_title)
            # store matches
            if len(match) != 0:
                if cur_year == None:
                    if (3000 in total_matches):
                        total_matches[3000].append(i)
                    else:
                        total_matches[3000] = [i]
                else:
                    if (cur_year in total_matches):
                        total_matches[cur_year].append(i)
                    else:
                        total_matches[cur_year] = [i]

        # sort returns by year
        od = collections.OrderedDict(sorted(total_matches.items()))
        matches_list = []
        for k, v in od.items():
            for n in v:
                matches_list.append(n)
        return matches_list

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.

        Example:
          sentiment = chatbot.extract_sentiment(
              chatbot.preprocess('I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """

        pat_neg = 'never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint|n\'t'
        pat_punc = '[.:;!?]'
        super_words_neg = ['hate', 'terribl', 'horribl', 'ridicul']
        super_words_pos = ['love', 'amaz', 'awesom', 'incred']
        flag_neg = 1
        line = preprocessed_input.lower()
        titles = self.extract_titles(line)

        for t in titles:
            line = line.replace(t, '')

        line = re.sub('[\'\"]', '', line)
        line = [xx.strip() for xx in re.split('(\W)', line)]
        line = [xx for xx in line if xx != '']
        score = 0

        for xx in line:
            if re.match(pat_punc, xx):
                flag_neg = 1
            if re.match('r+e+a+l+y+', xx):
                flag_neg *= 10
            elif re.match(pat_neg, xx):
                flag_neg *= -1
            else:
                stemmed = self.p.stem(xx)
                if stemmed in super_words_pos:
                    score += 10 * flag_neg
                elif stemmed in super_words_neg:
                    score += -10 * flag_neg
                elif stemmed in self.sentiment:
                    score += self.sentiment[stemmed] * flag_neg

        print(score)
        if score <= -10:
            return -2
        if score >= 10:
            return 2
        elif score < 0:
            return -1
        elif score > 0:
            return 1
        else:
            return 0

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of pre-processed text
        that may contain multiple movies. Note that the sentiments toward
        the movies may be different.

        You should use the same sentiment values as extract_sentiment, described above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess('I liked both "Titanic (1997)" and "Ex Machina".'))
          print(
              sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie title,
          and the second is the sentiment in the text toward that movie
        """
        pat_neg = 'never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint|n\'t'
        pat_punc = '[.:;!?]'
        flag_neg = 1
        line = preprocessed_input.lower()

        titles = re.findall('"([^"]*)"', preprocessed_input)

        for t in titles:
            t = t.lower()
            t = re.sub('\([0-9]+\)', '', t)
            t = t.strip()
            line = line.replace(t, ' MOVIEPLACEHOLDER ')
        scores = []
        line = re.sub('[\'\"]', '', line)
        line = [xx.strip() for xx in re.split('(\W)', line)]
        line = [xx for xx in line if xx != '']
        score = 0
        movie = 0

        # if movie title(s) appears before the description
        flag = 0
        for i in range(len(line)):
            xx = line[i]
            if re.match(pat_punc, xx):
                flag_neg = 1
            elif re.match(pat_neg, xx):
                flag_neg = -1
            else:
                stemmed = self.p.stem(xx)
                if stemmed in self.sentiment:
                    score += self.sentiment[stemmed] * flag_neg

            if xx == 'MOVIEPLACEHOLDER' or (flag > 0 and re.match('[,.:;!?]', xx)):
                # if negative score
                if score < 0:
                    if flag > 0:
                        for i in range(flag):
                            scores.append((titles[movie], -1))
                            movie += 1
                    else:
                        scores.append((titles[movie], -1))
                        movie += 1
                    score = 0
                    flag = 0

                # if positive score
                elif score > 0:
                    if flag > 0:
                        for i in range(flag):
                            scores.append((titles[movie], 1))
                            movie += 1
                    else:
                        scores.append((titles[movie], 1))
                        movie += 1
                    score = 0
                    flag = 0

                # if not first movie, and prev word was negation, take negative of previous
                elif movie > 0 and (line[i-1] == "not" or line[i-1] == "but"):
                    score = scores[movie-1][1] * -1
                    if flag > 0:
                        for i in range(flag):
                            scores.append((titles[movie], score))
                            movie += 1
                    else:
                        scores.append((titles[movie], score))
                        movie += 1
                    score = 0
                    flag = 0

                # if not first movie, and prev movie wasn't neutral and there is and or, or previous was placeholder:
                elif movie > 0 and score != 0 and scores[movie-1][1] != 0 and (line[i-1] == "and" or line[i-1] == "or" or line[i-1] == "MOVIEPLACEHOLDER"):
                    score = scores[movie-1][1]
                    if flag > 0:
                        for i in range(flag):
                            scores.append((titles[movie], score))
                            movie += 1
                    else:
                        scores.append((titles[movie], score))
                        movie += 1
                    score = 0
                    flag = 0

                # if at placeholder, next char is ,.! and prev score, then take prev movie score
                elif xx == 'MOVIEPLACEHOLDER' and movie > 0 and scores[movie-1][1] != 0 and (i < (len(line) - 1)) and re.match('[,.:;!?]', line[i+1]):
                    score = scores[movie-1][1]
                    if flag > 0:
                        for i in range(flag):
                            scores.append((titles[movie], score))
                            movie += 1
                    else:
                        scores.append((titles[movie], score))
                        movie += 1
                    score = 0
                    flag = 0

                # if prev char is ,.! and we are at new placeholder, then add flag
                elif xx == 'MOVIEPLACEHOLDER' and (i == 0 or re.match('[,.:;!?]', line[i-1]) or line[i-1] == "and"):
                    flag += 1

                # if truly neutral
                else:
                    if flag > 0:
                        for i in range(flag):
                            scores.append((titles[movie], 0))
                            movie += 1
                    else:
                        scores.append((titles[movie], 0))
                        movie += 1
                    score = 0
                    flag = 0
        return scores

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.

        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
          than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
          tying for minimum edit distance to the given movie.

        Example:
          chatbot.find_movies_closest_to_title(
              "Sleeping Beaty") # should return [1656]

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
        """
        dists = {}

        for i in range(len(self.titles)):

            movie_title = self.titles[i][0]

            # print(movie_title)
            # get everything before (
            movie_title_edited = re.findall('(.*) \(', movie_title)

            # print(movie_title_edited)
            # take away space

            # print(movie_title_edited)

            if len(movie_title_edited) != 0:
                movie_title_edited = movie_title_edited[0]

            else:
                movie_title_edited = movie_title

            # print(movie_title_edited)

            # print(movie_title_edited)
            movie_title_lc = movie_title_edited.lower()
            movie_title_lc = movie_title_lc

            # print(movie_title_lc)

            title_lc = title.lower()

            # print(title_lc)

            edit_dist = edit_distance(
                movie_title_lc, title_lc, substitution_cost=2, transpositions=False)

            # if movie_title_lc == "sleeping beauty":
            #     print(edit_dist)
            #     print("found movie")

            if edit_dist <= max_distance:

                if edit_dist not in dists:
                    dists[edit_dist] = list()

                dists[edit_dist].append(i)

        # print(dists)

        if not dists:  # dict is empty
            return []

        else:
            return dists[min(dists)]

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)

        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification

        Additional Notes:
        - clarification of "2" for all the Harry Potter Movies returns both the second entry and Deathly Hallows: Part 2
          > don't seem to need to separate "2" as second entry vs. common knowledge sequel because sorted returns by release year
        """
        total_matches = set([])
        clarification = clarification.lower()

        candidate_dict = {}

        # disambiguate (part 2): when the clarification is part of a movie title: sequel number or name, year, etc.

        max_query_overlap = 0
        max_overlap_indices = []
        for index in candidates:
            cur_title = self.titles[index][0].lower()
            cur_year = None

            # find the thing that matches ([0123456789]{4})
            # matches = re.search("\(([0123456789]{4})\)", cur_title)
            # print(year_start.group(), cur_title)
            pat = re.compile("\([0123456789]{4}\)")
            for match in pat.finditer(cur_title):
                cur_year = match.group()[1:-1].strip()
                cur_title = cur_title[:match.start()].strip()

            if cur_year == None:
                candidate_dict['3000'] = index
            else:
                candidate_dict[cur_year] = index

            # check if the clarification matches the year
            if clarification == cur_year:
                total_matches.add(index)

            # check if the phrase is in candidate names

            # prevent matching "one" to "prisoner"
            # but still allow multi-word clarifications to match
            # this one is checking for an exact match and we'll do approximate match later on
            # elif cur_title.find(clarification) >= 0:
                # print(cur_title, cur_title.find(clarification), clarification)

            matching = False
            cur_title_split = re.findall(r"[\w']+", cur_title)
            clarification_split = re.findall(r"[\w']+", clarification)

            for i in range(len(cur_title_split)):
                # if start of match found
                if clarification_split[0] == cur_title_split[i]:
                    matching = True
                    # check down the rest of the query
                    for j in range(len(clarification_split)):
                        if i+j < len(cur_title_split):
                            if clarification_split[j] != cur_title_split[i+j]:
                                matching = False
                        if i+j == len(cur_title_split):
                            # not the whole query was found. approximate can deal
                            matching = False
                    if (matching):
                        total_matches.add(index)
                        break

            # extraneous word substrings
            """
            Goal: Handle words but still match substring
            Functionality:
             - finds the candidate titles with the max overlap with query (can be multiple)
             - excludes prepositions, articles, "and"'s etc as viable for match using NLTK
             - requires exact matches between query word and title words, so for example
               it prevents "prisoner" from matching "one". (Could extend to use PorterStemmer)
            """
            count_query_overlap = 0
            list_of_words = word_tokenize(clarification)
            tagged_tokens = nltk.pos_tag(list_of_words)
            tagged = {x: y for (x, y) in tagged_tokens}
            pos_ignore = ['DT', 'CC', 'IN', 'TO', 'AT']

            # excluding split on ' bc word_tokenize doesn't either
            for word in list_of_words:
                # this does have the PorterStemmer issue, but...
                if word in cur_title_split and tagged[word] not in pos_ignore:
                    count_query_overlap += 1
            if count_query_overlap > max_query_overlap:
                max_query_overlap = count_query_overlap
                max_overlap_indices = [index]
            if count_query_overlap == max_query_overlap:
                max_overlap_indices.append(index)

        # disambiguate (part 3): location in the list of options, extraneous words, etc.

        # finish up extraneous word substring matching
        if max_query_overlap > 0:
            for item in max_overlap_indices:
                total_matches.add(item)

        # clarification is the index in the list of candidates ("2", "second", "2nd", "last")
        num_mapping = [['1', 'first', '1st'], ['2', 'second', '2nd', 'two', 'ii'], ['3', 'third', 'third', 'three', 'iii'], ['4', 'fourth', '4th', 'four', 'iv'], [
            '5', 'fifth', '5th', 'five', 'v'], ['6', 'sixth', '6th', 'six', 'vi'], ['7', 'seventh', '7th', 'seven', 'vii'], ['8', 'eighth', '8th', 'eight', 'viii'], ['9', 'ninth', '9th', 'nine', 'ix']]

        # remove extraneous words from this too
        for query_num in re.findall(r"[\w']+", clarification):
            values = [i for i, item in enumerate(
                num_mapping) if query_num in item]
            if len(values) == 1:
                total_matches.add(candidates[values[0]])
        # special rules for 'one' (not dealing with 'i')
        if clarification == "one" or clarification == "number one":
            total_matches.add(candidates[0])
        if clarification.find("last") >= 0:
            total_matches.add(candidates[len(candidates) - 1])

        # function tested in isolation, so need to sort by order year for chronology ones even tho this is included in find_movies_by_title
        # "last" ("last" vs "the last one" vs. "last of the mohecans" where it's part. last case doesn't seem like a problem)
        # clarification is chronological ("most recent", "oldest", "newest", etc.)
        # list of candidates is sorted by year
        # "second most recent" will be handled by "second" and not trip either of these
        od = collections.OrderedDict(sorted(candidate_dict.items()))
        candidate_chronology = [v for k, v in od.items()]
        if clarification.find("newest") >= 0 or clarification == "most recent" or clarification == "the most recent":
            total_matches.add(
                candidate_chronology[len(candidate_chronology) - 1])

        if clarification.find("oldest") >= 0:
            total_matches.add(candidate_chronology[0])

        return list(total_matches)

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive

        :returns: a binarized version of the movie-rating matrix
        """
        #############################################################################
        # TODO: Binarize the supplied ratings matrix. Do not use the self.ratings   #
        # matrix directly in this function.                                         #
        #############################################################################

        # The starter code returns a new matrix shaped like ratings but full of zeros.
        binarized_ratings = np.zeros_like(ratings)
        for i in range(binarized_ratings.shape[0]):
            for j in range(binarized_ratings.shape[1]):
                if ratings[i][j] != 0:
                    if ratings[i][j] > threshold:
                        binarized_ratings[i][j] = 1
                    else:
                        binarized_ratings[i][j] = -1

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################
        similarity = 0
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
          in descending order of recommendation
        """

        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        # Do not use the self.ratings matrix directly in this function.                       #
        #                                                                                     #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################

        nonzero_values = np.nonzero(user_ratings)[0]

        r_xis = np.zeros(ratings_matrix.shape[0])

        for user_choice in nonzero_values:

            for i in range(ratings_matrix.shape[0]):

                user_row = ratings_matrix[user_choice, ]
                row = ratings_matrix[i, ]

                if np.all(row == 0) or np.all(user_row == 0):
                    sim = 0

                else:

                    sim = np.dot(user_row, row)

                    nonzero_square = np.square(user_row)
                    nonzero_sum = np.sum(nonzero_square)
                    nonzero_sqrt = np.sqrt(nonzero_sum)

                    sim = sim / nonzero_sqrt

                    whole_square = np.square(row)
                    whole_sum = np.sum(whole_square)
                    whole_sqrt = np.sqrt(whole_sum)

                    sim = sim / whole_sqrt

                # Now have similarities, multiply by user rating down column.
                r_xi = sim * user_ratings[user_choice]

                r_xis[i] += r_xi

        # Check movie not in before adding to recommendations

        sorted_rxi = (-r_xis).argsort()

        recommendations = []

        index = 0

        while len(recommendations) < k:

            if user_ratings[sorted_rxi[index]] == 0:
                recommendations.append(sorted_rxi[index])
            index += 1

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return recommendations

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = 'debug info'
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6 instructions.
        Remember: in the starter mode, movie names will come in quotation marks and
        expressions of sentiment will be simple!
        Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
