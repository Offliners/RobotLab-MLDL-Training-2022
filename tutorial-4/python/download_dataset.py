import gdown
import os
import zipfile

dataset_id = '1ucESV_kFchF6ldW88xSNj2fxIj43yihz'

def download_unzip_file(id, output, output_path):
    gdown.download(id=id, output=output, quiet=False)
    
    if os.path.isfile(output):
        zf = zipfile.ZipFile(output, 'r')
        zf.extractall(output_path)
    else:
        print(f'{output} not found! Please check google drive!')


if __name__ == '__main__':
    os.makedirs('./data', exist_ok=True)

    output = 'data.zip'
    download_unzip_file(dataset_id, output, './')