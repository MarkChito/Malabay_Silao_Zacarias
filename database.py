from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

db = SQLAlchemy(app)

class Database(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    object_name = db.Column(db.String(255), nullable=False)
    
    def __repr__(self):
        return "<Name %r>" % self.id
    
    def insert(self, data):
        db.session.add(data)
        db.session.commit()

with app.app_context():
    db.create_all()