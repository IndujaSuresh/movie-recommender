from flask import Blueprint, render_template, redirect, url_for, request,flash
from . import db
from flask_login import login_required, current_user
from .models import User,Genre#,Search_history,Rating

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')


