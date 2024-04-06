from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

db = SQLAlchemy(app)

class Unlisted_Images(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    object_name = db.Column(db.String(255), nullable=False)
    
    def __repr__(self):
        return f"<Unlisted_Images id={self.id}>"
    
    @classmethod
    def insert(cls, data):
        db.session.add(data)

        db.session.commit()

    @classmethod
    def update(cls, old_name, new_name):
        data = cls.query.filter_by(object_name=old_name).all()
        
        for item in data:
            item.object_name = new_name
        
        db.session.commit()

    @classmethod
    def delete(cls, folder_name):
        data = cls.query.filter_by(object_name=folder_name).all()
        
        for item in data:
            db.session.delete(item)
        
        db.session.commit()
    
    @classmethod
    def is_record_available(cls, folder_name):
        data = cls.query.filter_by(object_name=folder_name).first()
        
        if data:
            return True
        
        return False

class Contact_Us_Messages(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    subject = db.Column(db.String(255), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    
    def __repr__(self):
        return "<Contact_Us_Messages %r>" % self.id
    
    @classmethod
    def insert(cls, data):
        db.session.add(data)

        db.session.commit()

class Newsletter_List(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    email = db.Column(db.String(255), nullable=False)
    
    def __repr__(self):
        return "<Newsletter_List %r>" % self.id
    
    @classmethod
    def insert(cls, data):
        db.session.add(data)

        db.session.commit()

with app.app_context():
    db.create_all()