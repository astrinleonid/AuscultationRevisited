import os

def clear_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("Folder does not exist:", folder_path)
        return

    # List all files and remove each one
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Removes files and symbolic links
            elif os.path.isdir(file_path):
                # Optionally, if you want to remove subdirectories, use:
                # shutil.rmtree(file_path)
                pass  # Currently skipping directories
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    os.rmdir(folder_path)
