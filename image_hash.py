import json
import hashlib

class Image_Hash():
    def hash(file_path):
        hasher = hashlib.sha256()

        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)
                
                if not data:
                    break
                
                hasher.update(data)
        return hasher.hexdigest()

    def load(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    def verify(image_hash, hash_dict):
        return image_hash in hash_dict.values()