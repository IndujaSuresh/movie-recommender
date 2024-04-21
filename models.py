from flask_login import UserMixin
from . import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, db.Sequence('seq_id', start=7046, increment=1),
               primary_key=True) 
    #id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100))
    
class Genre(UserMixin,db.Model):
    id = db.Column(db.Integer, primary_key=True)
    genres=db.Column(db.String(100))
    
#class Rating(UserMixin,db.Model):
#   slno=db.Column(db.Integer, primary_key=True)
#   userid= db.Column(db.Integer)
#   movieid = db.Column(db.Integer)
#   rating = db.Column(db.Integer)
    
#class Search_history(UserMixin,db.Model):
#    slno=db.Column(db.Integer, primary_key=True)
#    userid = db.Column(db.Integer)
 #   movie=db.Column(db.String(100))
