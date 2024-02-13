import os
import shutil

source_folder = 'mgh_fetal_202305/hast_mgz'
destination_folder = 'new_dataset'

# List of folders to keep
folders_to_keep = [
    f'mom_{str(i).zfill(3)}' for i in range(1, 403)
]


for folder in os.listdir(source_folder):
    if folder in folders_to_keep:
        source_path = os.path.join(source_folder, folder)
        dest_path = os.path.join(destination_folder, folder)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for subfolder in os.listdir(source_path):
            subfolder_path = os.path.join(source_path, subfolder)
            if os.path.isdir(subfolder_path):
                week_info = subfolder.split('_')
                weeks = None
                for info in week_info:
                    if info.endswith('w') and info[:-1].isdigit():
                        weeks = int(info[:-1])
                        break

                if weeks is not None and 13 < weeks < 26:
                    for root, dirs, files in os.walk(subfolder_path):
                        for file in files:
                            if file.endswith('.mgz'):
                                file_name, file_ext = os.path.splitext(file)
                                new_file_name = f"{weeks}_week_{file_name}{file_ext}"
                                shutil.copy(os.path.join(root, file), os.path.join(dest_path, new_file_name))



