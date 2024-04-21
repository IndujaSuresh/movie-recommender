import os
import csv
from flask import Blueprint, render_template, redirect, url_for, request,flash,Flask,send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User,Genre#,Rating,Search_history
from . import db
from flask_login import login_user, logout_user, login_required,current_user
from pandas import DataFrame, read_csv
import json
import sqlalchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import surprise
from surprise.reader import Reader
from surprise import Dataset
from surprise import SVD
from collections import defaultdict 
import pandas as pd
import numpy as np


auth = Blueprint('auth', __name__)

movies = pd.read_csv('moviessmall.csv')
Final = pd.read_csv('Final.csv')

movies=movies.drop("Unnamed: 0",axis=1)
movies=movies.drop("imdbId",axis=1)

Final=Final.drop("Unnamed: 0",axis=1)
Final=Final.drop("imdbId",axis=1)


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])

ratings = pd.read_csv('ratingssmall.csv')
ratings =ratings.drop("Unnamed: 0",axis=1)

movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title'])


def avg_rating():
    ratings = pd.read_csv('ratingssmall.csv')
    ratings=ratings.drop("Unnamed: 0",axis=1)
    avg =pd.merge(ratings,movies,on='movieId',)
    avg=pd.DataFrame(avg.groupby('title')['rating'].mean())
    avg = pd.merge(avg,movies, on='title', how='outer')
    avg['rating'] = avg['rating'].fillna(0)
    avg = [avg["movieId"], avg["rating"]]
    headers = ["movieId", "avg"]
    avg = pd.concat(avg, axis=1, keys=headers)
    return avg

#--------------
#preference recommendation------------------------
def Convert(string):
    li = list(string.split(" "))
    return li

def preference():
    gen = Genre.query.filter_by(id = current_user.id).first()
    pre = gen.genres
    lst = Convert(pre)
    c = len(lst)
    p={}
    for i in range(0, c):
        preference = movies.loc[movies['genres'].str.contains(lst[i], case=False)]
        preference = preference.sample(n=10)
        p[i] = preference.values
        p[i] =p[i].tolist()
    return lst,c,p

#--------------
#collaborative filtering----------------------------------
def model():
    #model based
    ratings = pd.read_csv('ratingssmall.csv')
    ratings=ratings.drop("Unnamed: 0",axis=1)
    np.random.seed(42) # replicating results
    reader = Reader(rating_scale=(0.5, 5)) #line_format by default order of the fields
    data = Dataset.load_from_df(ratings[["userId",  "movieId",  "rating"]], reader=reader)
    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()
    algo_SVD = SVD(n_factors = 11)
    algo_SVD.fit(trainset)
    # Predict ratings for all pairs (i,j) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo_SVD.test(testset)
    return predictions
    #model based end

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def get_top_n(predictions, userId, movies_df, ratings_df, n = 10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    #2. Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_n[uid] = user_ratings[: n ]
    
    #3. Tells how many movies the user has already rated
    user_data = ratings_df[ratings_df.userId == (userId)]
    print('User {0} has already rated {1} movies.'.format(userId, user_data.shape[0]))
    #4. Data Frame with predictions. 
    preds_df = pd.DataFrame([(id, pair[0],pair[1]) for id, row in top_n.items() for pair in row],
                        columns=["userId" ,"movieId","rat_pred"])
    #5. Return pred_usr, i.e. top N recommended movies with (merged) titles and genres. 
    pred_usr = preds_df[preds_df["userId"] == (userId)].merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId')
            
    #6. Return hist_usr, i.e. top N historically rated movies with (merged) titles and genres for holistic evaluation
    hist_usr = ratings_df[ratings_df.userId == (userId) ].sort_values("rating", ascending = False).merge\
    (movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId')
    
    return hist_usr, pred_usr


#------------------------------
#content based-----------------------------------------
def recommend_similar(title):

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index = Final.index.tolist())
    svd = TruncatedSVD(n_components=200)
    latent_matrix = svd.fit_transform(tfidf_df)
    n=200
    latent_matrix_1_df = pd.DataFrame(latent_matrix[:,0:n], index= Final.title.tolist())
    #take the latent vectors for a selected movie from both content and collaborative matrices
    a_1 = np.array(latent_matrix_1_df.loc[title]).reshape(1, -1)
    #calculate the similarity of this movie with other in the list
    score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
    #form a dataframe of similar movies
    dictDf = {'content': score_1 }# 'collaborative': score_2, 'hybrid': hybrid}
    similar = pd.DataFrame(dictDf, index = latent_matrix_1_df.index )
    #sort it on the basis of either content ,collaborative or hybrid,
    similar.sort_values('content', ascending=False, inplace=True)
    similar = pd.DataFrame(similar)
    similar['title']=similar.index
    result = pd.merge(similar,movies, on='title')
    result = pd.DataFrame(result)
    return result


#-------------------
#top rated-----------------------------------------
def top_rating():
    ratings = pd.read_csv('ratingssmall.csv')
    ratings=ratings.drop("Unnamed: 0",axis=1)
    ## Top rated movies
  
    top_data =pd.merge(ratings,movies,on='movieId')
    trend=pd.DataFrame(top_data.groupby('title')['rating'].mean())
    trend['total number of ratings'] = pd.DataFrame(top_data.groupby('title')['rating'].count()) 
    top_data =top_data.groupby('title')['rating'].mean().sort_values(ascending=False)
    top_rated=pd.merge(top_data,movies,on='title')
    top_rated =top_rated[1:11]
    top = pd.DataFrame(top_rated)
    return top        

#login------------------------------------------------

@auth.route('/login')
def login():
    return render_template('login.html')   

@auth.route('/login', methods=['POST'])
def login_post():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
               # flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('auth.profile'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')

    return render_template("login.html", user=current_user)
    


#signup-------------------------------------------------
@auth.route('/signup')
def signup():
    return render_template('signup.html')
    
@auth.route('/signup', methods=['POST'])
def signup_post():
 if request.method == 'POST':
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')
    confirm = request.form.get('confirm')
    user = User.query.filter_by(email=email).first() # if this returns a user, then the email already exists in database

    if user: # if a user is found, we want to redirect back to signup page so user can try again
        flash('Email address already exists')
        return redirect(url_for('auth.signup'))
        
    elif password != confirm:
        flash('Passwords dont match.', category='error') 
    # create a new user with the form data. Hash the password so the plaintext version isn't saved.
    else:
        new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

    # add the new user to the database
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('auth.genre'))
        
    return render_template("signup.html", user=current_user)   



#genre---------------------------------------------------
@auth.route('/genre')
def genre():
    return render_template('genre.html') 

@auth.route('/genre', methods=['POST'])
def genre_post():
    genres=""
    for x in request.form.getlist('mycheckbox'):
          genres+=x
          genres+=" "
    new_user = Genre(genres=genres, id=current_user.id)
    
    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()
    
    return redirect(url_for('auth.profile'))
    
    
@auth.route('/genre_u')
def genre_u():
    return render_template('genre_u.html')    

@auth.route('/genre_u', methods=['POST'])
def g_update():
    if request.method =='POST':
        id1 =current_user.id
        new_genres=""
        for x in request.form.getlist('mycheckbox'):
            new_genres+=x
            new_genres+=" "
        
        user = Genre.query.filter_by(id=id1).first()
        
        user.genres = new_genres
        db.session.commit()
       
        return redirect(url_for('auth.x'))


@auth.route('/x')
def x():
    pre,c,p = preference()
    return render_template('x.html', pre=pre, c=c, p=p) 



#forgot-----------------------------------------------------
@auth.route('/forgot')
def forgot():
    return render_template('forgot.html') 

@auth.route('/forgot', methods=['POST'])
def update():
        email = request.form.get('email')
        new_password = request.form.get('password')
        confirm = request.form.get('confirm')
        
        user = User.query.filter_by(email=email).first()
        if user:
            if (new_password == confirm):
                new_password = generate_password_hash(new_password, method='sha256')
                user.password = new_password
                db.session.commit()
        #return render_template('forgot.html') 
                return render_template('login.html')
            else:
                flash('Passwords dont match.', category='error') 
                return redirect(url_for('auth.update'))
        else:
             flash('Email address dont exists')
             return render_template('signup.html')   
        
   
    

#rating--------------------------------------------------------    
@auth.route('/rating')
def rating():
    return render_template('rating.html') 
    
@auth.route('/rating', methods=['POST'])
def rating_post():
    
    ratings = pd.read_csv('ratingssmall.csv')
    ratings=ratings.drop("Unnamed: 0",axis=1)

    index= ratings.index.stop+1
    rating = request.form.get('rating')
    u_id = current_user.id
    m_id = request.form.get('movieId')
    row = [index,u_id,m_id,rating]
    if ((((ratings['userId'] == u_id) & (ratings['movieId'] == m_id)).any()) == True):
        i = ratings[(ratings['userId'] == u_id) & (ratings['movieId'] == m_id)].index
        ratings.drop(i, inplace = True) 
    else:    
        append_list_as_row('ratingssmall.csv', row)
    return redirect(url_for('auth.profile'))
   
#profile--------------------------------------------------------
@auth.route('/profile')
@login_required
def profile():
    try:
        
        top = top_rating()

        mid = top['movieId'].values.tolist()
        title= top['title'].values.tolist()
        tmdb = top['tmdbId'].values.tolist()
        year = top['year'].to_list()
        genres = top['genres'].to_list()

        return render_template("profile.html", len=len(top), movies=top, movieid=mid, movie_name=title ,tmdb=tmdb, year=year, genres=genres,name=current_user.name)
    except Exception as e:
        return str(e)
        
        
@auth.route('/profile', methods=['POST'])
def main():
        all_titles = movies['title'].tolist()
        if request.method == 'POST':
            m_name = request.form['movie_name']
            m_name = m_name.title()
            if m_name not in all_titles:
                return render_template('negative.html', name=m_name)
        #if request.method == 'POST':
           # m_name = request.form['movie_name']
           # titles = all_titles()
           # if m_name not in titles:
           #     return render_template('negative.html')
            else:
                details = movies.loc[movies['title'] == m_name]
                tmdb = details['tmdbId'].to_list()
            
                content = recommend_similar(m_name)
                content = content.head(7)
                movie_name = content['title'].to_list()
                movieId    = content['tmdbId'].to_list()
                movie_year = content['year'].to_list()
                genres  = content['genres'].to_list()
                m_id = content['movieId'].to_list()
                

                return render_template('positive.html',  name=m_name, len =len(content), tmdb=tmdb, movie_name=movie_name, movieId=movieId, movie_year=movie_year, movie_genre=genres, m_id=m_id)
 


#positive---------------------------------------------------------
@auth.route('/positive',methods=['POST'])
def positive():
    if request.method == 'POST':
        m_id = request.form.get('movieId')
        name = request.form.get('name')
        return render_template("rating.html", m_id=m_id, name=name)


#history------------------------------------------------------------
@auth.route('/usr_profile')
def usr_profile():
    ratings = pd.read_csv('ratingssmall.csv')
    ratings=ratings.drop("Unnamed: 0",axis=1)
    predictions = model()
    history, pred =get_top_n(predictions, movies_df = movies, userId = current_user.id, ratings_df = ratings)
    movie_name = history['title'].values.tolist()
    tmdb = history['tmdbId'].values.tolist()
    genres = history['genres'].values.tolist()
    year = history['year'].values.tolist()

    return render_template('usr_profile.html', len=len(history), movie_name=movie_name, tmdb=tmdb, year=year, genres=genres)   

#movies_u_may_like-------------------------------------------------------
@auth.route('/movies_u_may_like')
def movies_u_may_like():
    ratings = pd.read_csv('ratingssmall.csv')
    ratings=ratings.drop("Unnamed: 0",axis=1)
    predictions = model()
    history, pred = get_top_n(predictions, movies_df = movies, userId = current_user.id, ratings_df = ratings)
    avg = avg_rating()
    pred =pd.merge(avg,pred,on='movieId')
    avg_rate = pred['avg'].values.tolist()
    movie_name = pred['title'].values.tolist()
    tmdb = pred['tmdbId'].values.tolist()
    year = pred['year'].values.tolist()
    genres = pred['genres'].values.tolist()

    return render_template('movies_u_may_like.html', len=len(pred), his=len(history), movie_name=movie_name, tmdb=tmdb, year=year, genres=genres, avg=avg)

 
  

@auth.route('/negative')
def negative():
    return render_template("negative.html")        
  
  

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))
