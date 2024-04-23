import os
import json
import hashlib

class Image_Hash:
    @classmethod
    def hash(cls, file_path):
        hasher = hashlib.sha256()

        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)
                
                if not data:
                    break
                
                hasher.update(data)
        return hasher.hexdigest()

    @classmethod
    def hash_images_in_folder(cls, folder_path):
        image_hashes = {}

        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_hash = cls.hash(file_path)
                    image_hashes[file_path] = image_hash
        
        return image_hashes

    @classmethod
    def save(cls, image_hashes, json_file):
        with open(json_file, 'w') as f:
            json.dump(image_hashes, f)

    @classmethod
    def load(cls, json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    @classmethod
    def verify(cls, image_hash, hash_dict):
        return image_hash in hash_dict.values()