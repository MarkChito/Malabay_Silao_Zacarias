import json
import os

class Session:
    def __init__(self, file_path='static/session/session.json'):
        self.file_path = file_path
        self.session_data = self.__load_session_data__()

    def __load_session_data__(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    return {}
        else:
            return {}

    def __save_session_data__(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.session_data, file, indent=4)

    def set(self, session_name, value):
        self.session_data[session_name] = value
        self.__save_session_data__()

    def get(self, session_name):
        return self.session_data.get(session_name)

    def unset(self, session_name):
        try:
            del self.session_data[session_name]
            self.__save_session_data__()
        except:
            pass