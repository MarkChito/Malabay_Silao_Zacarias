from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import bcrypt

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

db = SQLAlchemy(app)

class Unlisted_Images(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, nullable=False)
    object_name = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255), nullable=False)
    
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

class User_Accounts(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    name = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    user_type = db.Column(db.String(255), nullable=False)
    
    def __repr__(self):
        return "<User_Accounts %r>" % self.id
    
    @classmethod
    def password_hash(self, password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    @classmethod
    def password_verify(self, password, hashed_password):
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

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
    def is_record_available(cls, username):
        data = cls.query.filter_by(username=username).first()
        
        if data:
            return data
        
        return False
    
class Upload_History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, nullable=False)
    image_name = db.Column(db.String(255), nullable=False)
    church_name = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255), nullable=False)
    building_capacity = db.Column(db.String(255), nullable=False)
    date_built = db.Column(db.String(255), nullable=False)
    short_description = db.Column(db.Text, nullable=False)
    
    def __repr__(self):
        return "<Upload_History %r>" % self.id
    
    @classmethod
    def insert(cls, data):
        db.session.add(data)

        db.session.commit()

    @classmethod
    def admin_select(cls):
        data = cls.query.all()
        
        if not data:
            return False
        
        return data
    
    @classmethod
    def user_select(cls, user_id):
        data = cls.query.filter_by(user_id=user_id).all()
        
        if not data:
            return False
        
        return data

class Church_Details(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_created = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    church_code = db.Column(db.String(255), nullable=False)
    church_name = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255), nullable=False)
    building_capacity = db.Column(db.String(255), nullable=False)
    date_built = db.Column(db.String(255), nullable=False)
    short_description = db.Column(db.Text, nullable=False)
    
    def __repr__(self):
        return "<Church_Details %r>" % self.id
    
    @classmethod
    def insert(cls, data):
        db.session.add(data)

        db.session.commit()

    @classmethod
    def is_record_available(cls):
        data = cls.query.all()
        
        if data:
            return data
        
        return False
    
    @classmethod
    def select(cls, church_code):
        data = cls.query.filter_by(church_code=church_code).first()
        
        if data:
            return data
        
        return False

def insert_admin_data():
    if not User_Accounts.is_record_available('admin'):
        hashed_password = User_Accounts.password_hash("admin123")

        admin_user = User_Accounts(name='Administrator', username='admin', password=hashed_password, user_type='admin')
        
        User_Accounts.insert(admin_user)

def insert_church_details():
    if not Church_Details.is_record_available():
        church_dictionary = {
            "barasoain_church_malolos_bulacan": {
                "church_code": "barasoain_church_malolos_bulacan",
                "church_name": "Barasoain Church (Malolos, Bulacan)",
                "location": "Malolos, Bulacan",
                "building_capacity": "Approximately 1,000 people",
                "date_built": "1885",
                "short_description": "Historic church known for its role in Philippine history, particularly during the Malolos Congress in 1898.",
            },
            "daraga_church_daraga_albay": {
                "church_code": "daraga_church_daraga_albay",
                "church_name": "Daraga Church (Daraga, Albay)",
                "location": "Daraga, Albay",
                "building_capacity": "Approximately 500 people",
                "date_built": "1773",
                "short_description": "Baroque-style church with picturesque views of Mayon Volcano.",
            },
            "manaoag_church_manaoag_pangasinan": {
                "church_code": "manaoag_church_manaoag_pangasinan",
                "church_name": "Manaoag Church (Manaoag, Pangasinan)",
                "location": "Manaoag, Pangasinan",
                "building_capacity": "Approximately 2,000 people",
                "date_built": "1610",
                "short_description": "Famous pilgrimage site housing the Our Lady of Manaoag, believed to be miraculous.",
            },
            "manila_cathedral_church_intramuros_manila": {
                "church_code": "manila_cathedral_church_intramuros_manila",
                "church_name": "Manila Cathedral Church (Intramuros, Manila)",
                "location": "Intramuros, Manila",
                "building_capacity": "Approximately 800 people",
                "date_built": "1581 (original construction), rebuilt several times since then",
                "short_description": "One of the oldest churches in the Philippines, seat of the Archdiocese of Manila.",
            },
            "paoay_church_paoay_ilocos_norte": {
                "church_code": "paoay_church_paoay_ilocos_norte",
                "church_name": "Paoay Church (Paoay, Ilocos Norte)",
                "location": "Paoay, Ilocos Norte",
                "building_capacity": "Approximately 1,000 people",
                "date_built": "1704",
                "short_description": "UNESCO World Heritage Site known for its distinct architecture featuring earthquake-resistant coral stone.",
            },
            "quiapo_church_quiapo_manila": {
                "church_code": "quiapo_church_quiapo_manila",
                "church_name": "Quiapo Church (Quiapo, Manila)",
                "location": "Quiapo, Manila",
                "building_capacity": "Approximately 10,000 people",
                "date_built": "1586 (original construction), rebuilt in 1933",
                "short_description": "Famous for the Black Nazarene, a dark-skinned statue of Jesus believed to be miraculous.",
            },
            "san_agustin_church_intramuros_manila": {
                "church_code": "san_agustin_church_intramuros_manila",
                "church_name": "San Agustin Church (Intramuros, Manila)",
                "location": "Intramuros, Manila",
                "building_capacity": "Approximately 700 people",
                "date_built": "1587",
                "short_description": "Oldest stone church in the Philippines, a UNESCO World Heritage Site.",
            },
            "santa_maria_church_santa_maria_ilocos_sur": {
                "church_code": "santa_maria_church_santa_maria_ilocos_sur",
                "church_name": "Santa Maria Church (Santa Maria, Ilocos Sur)",
                "location": "Santa Maria, Ilocos Sur",
                "building_capacity": "Approximately 1,000 people",
                "date_built": "1765",
                "short_description": "Declared as a UNESCO World Heritage Site, featuring a distinct bell tower separate from the main church building.",
            },
            "taal_basilica_church_taal_batangas": {
                "church_code": "taal_basilica_church_taal_batangas",
                "church_name": "Taal Basilica Church (Taal, Batangas)",
                "location": "Taal, Batangas",
                "building_capacity": "Approximately 2,000 people",
                "date_built": "1575 (original construction), rebuilt in 1755 after destruction from Taal Volcano eruption",
                "short_description": "One of the largest Catholic churches in the Philippines, known for its Baroque architecture.",
            },
            "miagao_church_miagao_iloilo": {
                "church_code": "miagao_church_miagao_iloilo",
                "church_name": "Miagao Church (Miagao, Iloilo)",
                "location": "Miagao, Iloilo",
                "building_capacity": "Approximately 800 people",
                "date_built": "1786",
                "short_description": "Declared as a UNESCO World Heritage Site, known for its fortress-like appearance and intricate facade.",
            }
        }

        for _, church_info in church_dictionary.items():
            data = Church_Details(church_code=church_info["church_code"], church_name=church_info["church_name"], location=church_info["location"], building_capacity=church_info["building_capacity"], date_built=church_info["date_built"], short_description=church_info["short_description"])
        
            Church_Details.insert(data)

with app.app_context():
    db.create_all()

    insert_admin_data()
    insert_church_details()