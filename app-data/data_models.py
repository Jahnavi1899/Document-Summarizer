class FileModelManage:
    def __init__(self):
        self.file_model_map = {}

    def add_file_entry(self, filename):
        """Initialize a new file entry if it doesn't exist."""
        if filename not in self.file_model_map:
            self.file_model_map[filename] = {}

    def update_file_info(self, filename, key, value):
        """Update or add a specific piece of information for a file."""
        if filename not in self.file_model_map:
            self.add_file_entry(filename)
        self.file_model_map[filename][key] = value

    def get_file_data(self, filename):
        """Retrieve all data for a specific file."""
        return self.file_model_map.get(filename)

    def get_file_info(self, filename, key):
        """Retrieve a specific piece of information for a file."""
        # print(f"Inside get_file_info method:{self.file_model_map}")
        file_data = self.get_file_data(filename)
        # print(f"Inside get_file_info method:{file_data}")
        return file_data.get(key) if file_data else None

    def list_files(self):
        """List all files in the map."""
        return list(self.file_model_map.keys())

    def remove_file_entry(self, filename):
        """Remove a file entry from the map."""
        if filename in self.file_model_map:
            del self.file_model_map[filename]
            return True
        return False