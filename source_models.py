import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")

import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")





import pandas as pd
import numpy as np

data = pd.read_csv("ratings_Beauty.csv")
import plotly.graph_objects as graph
duplicates = data.duplicated(["UserId", "ProductId", "Rating", "Timestamp"]).sum()
plot = graph.Figure([graph.Bar(x = data["Rating"].value_counts().index, y = list(data["Rating"].value_counts()), textposition = "auto")])
plot.update_layout(title_text = "rating count",xaxis_title = "Rating value",yaxis_title = "Number of ratings")


rated_users = data.groupby("UserId")["Rating"].count().sort_values(ascending = False)
rated_products = data.groupby("ProductId")["Rating"].count().sort_values(ascending = False)

user_products = data.groupby("ProductId")["UserId"].count().sort_values(ascending = False)

print("Number of users that rate each product:\n")
print(user_products)

import pandas as pd
import numpy as np

rated_products_df = pd.DataFrame(rated_products)

# Create a function to categorize ratings
def categorize_ratings(rating):
    if rating < 10:
        return "< 10"
    elif rating >= 10 and rating < 50:
        return "10-49"
    elif rating >= 50 and rating < 100:
        return "50-99"
    else:
        return ">= 100"

# Apply the function to the ratings column
rated_products_df["Category"] = rated_products_df["Rating"].apply(categorize_ratings)

# Count the occurrences of each category
category_counts = rated_products_df["Category"].value_counts()

# Print the results
print("Number of products with < 10 ratings: ", category_counts["< 10"])
print("Number of products with >= 10 and < 50 ratings: ", category_counts["10-49"])
print("Number of products with >= 50 and < 100 ratings: ", category_counts["50-99"])
print("Number of products with >= 100 ratings: ", category_counts[">= 100"])
print("Average number of products rated by users: ", rated_products_df["Rating"].mean())

x_axis = ["Number of products with < 10 ratings", "Number of products with >= 10 and < 50 ratings",
          "Number of products with >= 50 and < 100 ratings", "Number of products with >= 100 ratings"]
y_axis = [category_counts["< 10"], category_counts["10-49"], category_counts["50-99"], category_counts[">= 100"]]

plot = graph.Figure([graph.Bar(x = data["ProductId"].value_counts().nlargest(5).index, y = list(data["ProductId"].value_counts()), textposition = "auto")])

plot.update_layout(title_text = "The top 5 most popular products",
                   xaxis_title = "ProductId",
                   yaxis_title = "Number of occurence")

#plot.show()
# Encode alphanumerical data as numerical data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

encoded_data = data
encoded_data["User"] = label_encoder.fit_transform(data["UserId"])
encoded_data["Product"] = label_encoder.fit_transform(data["ProductId"])

print(encoded_data.head())

# Find the average ratings by each user
average_rating = encoded_data.groupby("User")["Rating"].mean()

# Merge it with the dataset
encoded_data = pd.merge(encoded_data, average_rating, on = "User")

# Rename the columns
encoded_data = encoded_data.rename(columns = {"Rating_x": "Original_rating", "Rating_y": "Average_rating"})

# Normalize the ratings
encoded_data["Normalized_rating"] = encoded_data["Original_rating"] - encoded_data["Average_rating"]

rated_products_encoded = encoded_data.groupby("Product")["Original_rating"].count()
rated_products_encoded_df = pd.DataFrame(rated_products_encoded)
filtered_rated_products = rated_products_encoded_df[rated_products_encoded_df.Original_rating >= 200]
popular_products = filtered_rated_products.index.tolist()
remaining_data = encoded_data[encoded_data["Product"].isin(popular_products)]

# Create user-item matrix
user_item_matrix = pd.pivot_table(remaining_data, values = "Normalized_rating", index = "UserId", columns = "Product")
user_item_matrix = user_item_matrix.fillna(0)

print(user_item_matrix.head(5))

# find k users that have the highest similarity to the chosen user
import operator
from sklearn.metrics.pairwise import cosine_similarity


def top_k_similar(user_id, user_item_matrix, k):
    selected_user = user_item_matrix.loc[user_id]
    remaining_users = user_item_matrix.drop(index=user_id)

    similarity_scores = cosine_similarity([selected_user], remaining_users)[0]

    user_similarity_mapping = dict(zip(remaining_users.index, similarity_scores))

    sorted_user_similarity = sorted(user_similarity_mapping.items(), key=operator.itemgetter(1), reverse=True)

    top_similar_users = [user[0] for user in sorted_user_similarity[:k]]

    return top_similar_users

# Find the top k users that are most similar to the chosen user

#assign the number of
k = 5

#assign a chosen user
users = user_item_matrix.index.tolist()
user_id = users[np.random.randint(0, user_item_matrix.shape[0] + 1)]

#calculate the k similarity
similar_users = top_k_similar(user_id, user_item_matrix, k)
print("The top {} users that are most similar to the chosen user are {}".format(k, similar_users))

# recommend k products to the chosen user
import pandas as pd

def top_m_products(user_id, similar_users, user_item_matrix, k):
    related_user_products = encoded_data[encoded_data.UserId.isin(similar_users)]
    related_user_ratings = user_item_matrix.loc[similar_users]

    related_user_avg = related_user_ratings.mean(axis=0)
    related_user_avg_df = pd.DataFrame(related_user_avg, columns=["Average"])

    target_user_ratings = user_item_matrix.loc[user_id]

    target_user_ratings_df = target_user_ratings.to_frame(name="Rating")
    unrated_products_indices = target_user_ratings_df[target_user_ratings_df["Rating"] == 0].index

    filtered_avg_ratings = related_user_avg_df[related_user_avg_df.index.isin(unrated_products_indices)]
    sorted_avg_ratings = filtered_avg_ratings.sort_values(by=["Average"], ascending=False)

    top_k_products = sorted_avg_ratings.head(k).index.tolist()

    return top_k_products


# recommend 5 products to our chosen user
def decode_product_ids(encoded_product_ids, label_encoder):
    return label_encoder.inverse_transform(encoded_product_ids)

# Get the encoded product values using the top_m_products function
encoded_product_ids = top_m_products(user_id, similar_users, user_item_matrix, k)

# Convert the encoded values back to the original ProductId
decoded_product_ids = decode_product_ids(encoded_product_ids, label_encoder)

# Print the results
print(f"The top {k} recommended products are: {', '.join(map(str, decoded_product_ids))}")









