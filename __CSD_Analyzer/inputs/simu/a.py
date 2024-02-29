import os

folder_path = 'original'  # 対象のフォルダパスを指定

# フォルダ内のファイルを取得
files = os.listdir(folder_path)

# ファイルごとに処理
for file in files:
    file_path = os.path.join(folder_path, file)

    if "_gray" in file:
        new_file_name = file.replace("_gray", "")
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(file_path, new_file_path)

    """
    # ファイル名に"_gray"が含まれていない場合は削除
    if "_gray" not in file:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    """

remaining_files = os.listdir(folder_path)
