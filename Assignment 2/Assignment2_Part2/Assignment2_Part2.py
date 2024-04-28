# FILENAME: Assignment2_Part2.py
# DUE DATE: 4/28/2024
# AUTHOR:   Reg Gonzalez
# EMAIL:    rdg170330@utdallas.edu (school) or regmckie@gmail.com (personal)
# COURSE:   CS 6375.001, Spring 2024
# VERSION:  1.0
#
# DESCRIPTION:
# Twitter provides a service for posting short messages. In practice, many of the tweets are very
# similar to each other and can be clustered together. By clustering similar tweets together, we can
# generate a more concise and organized representation of the raw tweets, which will be very
# useful for many Twitter-based applications (e.g., truth discovery, trend analysis, search ranking,
# etc.).
#
# In this assignment, you will learn how to cluster tweets by utilizing Jaccard Distance metric and
# K-means clustering algorithm.
#
# The objectives are to:
# Compute the similarity between tweets using the Jaccard Distance metric.
# Cluster tweets using the K-means clustering algorithm.

import math
import re
import random as rd

'''
Perform pre-processing on the tweets in the datafile.

These are the requirements for cleaning up the tweets: 
    - Remove the tweet ID and timestamp
    - Remove any word that starts with "@" (e.g., @Sidapa)
    - Remove any hashtag symbol (e.g., convert #necromancy to necromancy)
    - Remove any URL
    - Convert every word to lowercase
    
PARAMETERS:
    - datafile: the URL of the datafile we read in
    
RETURNS:
    - tweets: list of preprocessed tweets
'''
def preprocess_data(datafile):

    processed_tweets = []
    file = open(datafile, "r", encoding="utf-8")
    list_of_tweets = list(file)

    # Iterate through list of tweets and clean it up
    for counter in range(len(list_of_tweets)):

        # Start by leading and trailing whitespaces
        list_of_tweets[counter] = list_of_tweets[counter].strip()

        # Remove the tweet ID and timestamp
        # (The first 50 characters of each line contains this info)
        list_of_tweets[counter] = list_of_tweets[counter][50: ]

        # Remove any word that starts with "@" (e.g., @Sidapa)
        list_of_tweets[counter] = ' '.join(word for word in list_of_tweets[counter].split()
                                           if not word.startswith('@'))

        # Remove any hashtag symbol (e.g., convert #necromancy to necromancy)
        list_of_tweets[counter] = list_of_tweets[counter].replace('#', '')

        # Remove any URL
        url_pattern = re.compile(r'http?://\S+|https?://\S+|www\.\S+')
        list_of_tweets[counter] = re.sub(url_pattern, '', list_of_tweets[counter])

        # Remove any extra spaces
        list_of_tweets[counter] = " ".join(list_of_tweets[counter].split())

        # Convert every word to lowercase
        list_of_tweets[counter] = list_of_tweets[counter].lower()

        # Append newly processed tweet to new list
        processed_tweets.append(list_of_tweets[counter].split(' '))

    file.close()

    return processed_tweets


'''
Tests the K-means algorithm to see if it converged (i.e., the algorithm is complete)

PARAMETERS:
    - tweet_centroids: list of centroids
    - past_centroids: list of past centroids that were generated 

RETURNS:
    - convergence_occurred: returns True if convergence occurred and False otherwise
'''
def convergence(tweet_centroids, past_centroids):

    convergence_occurred = True

    # If length of two centroids lists aren't equal, this means that
    # convergence is not done—the algorithm is still going
    if len(tweet_centroids) != len(past_centroids):
        convergence_occurred = False
        return convergence_occurred

    # Convergence occurs when centroids stop updating, so check to see
    # if the lists are the same
    for counter in range(len(tweet_centroids)):
        if tweet_centroids[counter] != past_centroids[counter]:
            convergence_occurred = False
            return convergence_occurred

    return convergence_occurred


'''
Calculate Jaccard Distance—used to create clusters and determine centroids.

PARAMETERS:
    - processed_tweet: some tweet
    - tweet_centroid: some centroid (which is also a tweet)
    
RETURNS:
    - jaccardDistance: Jaccard Distance of the two tweets
'''
def calculateJaccardDistance(processed_tweet, tweet_centroid):

    # Jaccard Distance = 1 - (|A intersection B| / |A union B|)
    numerator = set(tweet_centroid).intersection(processed_tweet)
    denominator = set().union(tweet_centroid, processed_tweet)
    jaccardDistance = 1 - (len(numerator) / len(denominator))

    return jaccardDistance


'''
Calculate the sum of squared errors for each tweet in each cluster.

PARAMETERS:
    - tweet_clusters: dictionary of clusters
    
RETURNS:
    - sum_of_squared_errors: Sum of squared errors of the distance of tweet from its centroid
'''
def calculateSumOfSquaredErrors(tweet_clusters):

    sum_of_sqaured_errors = 0
    tweet_distance_index = 1    # In the 'tweet_clusters' dict, the tweet's distance is located at index 1

    # Go through each tweet in each cluster and calculate the sum of squared errors
    for cluster in range(len(tweet_clusters)):
        for tweet in range(len(tweet_clusters[cluster])):
            sum_of_sqaured_errors += (tweet_clusters[cluster][tweet][tweet_distance_index] *
                                      tweet_clusters[cluster][tweet][tweet_distance_index])

    return sum_of_sqaured_errors


'''
Create clusters for K-means algorithm.

PARAMETERS:
    - tweet_centroids: list of centroids
    - processed_tweets: list of tweets
    
RETURNS:
    - tweets_clusters: list of clusters generated
'''
def createClusters(tweet_centroids, processed_tweets):

    # Initialize dictionary to keep track of clusters
    # Key: cluster index
    # Value: list of tweets assigned to that cluster
    tweet_clusters = dict()

    # Assign a centroid to every tweet
    for i in range(len(processed_tweets)):
        # Initialize cluster index and distance, which will keep track
        # of the index of the closest centroid to the current tweet and the minimum distance
        tweet_cluster_index = -1
        distance = math.inf
        for j in range(len(tweet_centroids)):
            jaccard_distance = calculateJaccardDistance(processed_tweets[i], tweet_centroids[j])

            # In the occasion that the centroid is equal to the iterated tweet,
            # assign that tweet to be the centroid
            if tweet_centroids[j] == processed_tweets[i]:
                distance = 0
                tweet_cluster_index = j
                break

            # We want to minimize distance, so if the calculated Jaccard Distance is lower
            # than our current distance, update it
            if jaccard_distance < distance:
                distance = jaccard_distance
                tweet_cluster_index = j

        # Randomly assign tweet as centroid if need be
        if distance == 1:
            tweet_cluster_index = rd.randint(0, len(tweet_centroids) - 1)

        # To make clusters, assign each tweet the centroid that's closest to them.
        # If the index for the cluster doesn't exist, create a new list for it
        tweet_clusters.setdefault(tweet_cluster_index, []).append([processed_tweets[i]])

        # Each tweet is represented as a list: first element is the tweet itself and the
        # second element is the distance between that tweet and its closest centroid.
        distance_index = len(tweet_clusters.setdefault(tweet_cluster_index, [])) - 1
        tweet_clusters.setdefault(tweet_cluster_index, [])[distance_index].append(distance)

    return tweet_clusters


'''
Updates the centroids for each cluster with the tweet with the minimum distance.

PARAMETERS:
    - tweet_clusters: dictionary of clusters 
    
RETURNS:
    - tweet_centroids: list of centroids 
'''
def updateTweetCentroids(tweet_clusters):

    tweet_centroids = []

    # Go through each cluster
    for cluster in range(len(tweet_clusters)):
        minimum_distances_list = []         # Stores already calculated minimum distances
        tweet_centroid_index = -1           # Stores index of centroid
        minimum_sum_distances = math.inf    # Stores the minimum sum distance, which will be used to select centroid

        # For every tweet in the cluster, calculate the distance between it and
        # every other tweet in the cluster
        for one_tweet in range(len(tweet_clusters[cluster])):
            sum_distances = 0   # Stores sum of distances between one tweet & every other tweet in cluster
            minimum_distances_list.append([])   # For each tweet, append a list for minimum distances calculated

            for other_tweet in range(len(tweet_clusters[cluster])):
                # If 'other' tweet is the same as the one you're looking at, just store 0 for distance
                if one_tweet == other_tweet:
                    minimum_distances_list[one_tweet].append(0)
                # If 'other' tweet is not the same as the one you're looking at either:
                # (1) calculate the Jaccard Distance if it isn't already calculated OR
                # (2) or just get the already calculated distance from the list
                else:
                    if other_tweet < one_tweet:
                        distance = minimum_distances_list[other_tweet][one_tweet]
                    else:
                        distance = calculateJaccardDistance(tweet_clusters[cluster][other_tweet][0],
                                                                   tweet_clusters[cluster][one_tweet][0])

                    # Add newly calculated distance to sum of distances
                    # and append it to the already calculated distances list
                    sum_distances = sum_distances + distance
                    minimum_distances_list[one_tweet].append(distance)

            # Update minimum distance and index accordingly
            if sum_distances < minimum_sum_distances:
                tweet_centroid_index = one_tweet
                minimum_sum_distances = sum_distances

        # Assign newly updated centroid to tweet with the minimum distance
        tweet_centroids.append(tweet_clusters[cluster][tweet_centroid_index][0])

    return tweet_centroids


'''
Performs the K-means algorithm.

PARAMETERS:
    - processed_tweets: list of tweets
    - iterations: maximum number of iterations
    - K: value for K in algorithm; number of clusters we'll get in the end
    
RETURNS:
    - sum_of_squared_errors: sse resulted from creating clusters
    - tweet_clusters: the clusters we get at the end (# of clusters = K); a dictionary
'''
def KMeansAlgorithm(processed_tweets, K=5, iterations=50):

    counter = 0
    iterations_counter = 0
    past_centroids = []
    tweet_centroids = []
    tweets_used_for_centroids = dict()  # Used to store which tweets have already been used as centroids

    # Randomly initialize K random centroids
    while counter < K:
        # Index of a randomly chosen tween in the list of processed tweets
        tweet_index = rd.randint(0, len(processed_tweets) - 1)
        
        # Check if random tweet was already used as a centroid,
        # if not, then add that tweet to the list of tweets used for centroids
        if tweet_index not in tweets_used_for_centroids:
            counter = counter + 1
            tweets_used_for_centroids[tweet_index] = True
            tweet_centroids.append(processed_tweets[tweet_index])

    # Keep running the K-means algorithm until convergence or the total number
    # of iterations has reached its limit
    while (iterations_counter < iterations) and (convergence(tweet_centroids, past_centroids)) == False:

        # Create clusters
        tweet_clusters = createClusters(tweet_centroids, processed_tweets)

        # Centroids become past centroids as the algorithm continues
        past_centroids = tweet_centroids
        
        # Update centroids
        tweet_centroids = updateTweetCentroids(tweet_clusters)

        iterations_counter += 1

        # Calculate the sum of squared errors
        sum_of_squared_errors = calculateSumOfSquaredErrors(tweet_clusters)

    return sum_of_squared_errors, tweet_clusters


if __name__ == '__main__':

    # Read in the datafile and perform pre-processing
    datafile = "nytimeshealth.txt"
    processed_tweets = preprocess_data(datafile)

    K = 5
    no_of_runs = 5

    # Start with K = 5 (i.e., 5 clusters).
    # Per the instructions, we need to test five different values of K,
    # so go through a loop to test K = 5, 6, 7, 8, and 9.
    for run in range(no_of_runs):
        print("K-means for k = " + str(K) + ":")

        sum_of_squared_errors, tweet_clusters = KMeansAlgorithm(processed_tweets, K)

        for cluster in range(len(tweet_clusters)):
            no_of_tweets_in_cluster = str(len(tweet_clusters[cluster]))
            print("Cluster " + str(cluster + 1) + ": ", no_of_tweets_in_cluster + " tweets")

        print("Sum of squared errors: " + str(sum_of_squared_errors))
        print("\n")

        # Increment K (i.e., # of clusters) to next value
        K += 1