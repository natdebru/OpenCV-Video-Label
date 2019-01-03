from google_drive_downloader import GoogleDriveDownloader as gdd
import tarfile
import os


# allows user to download the re3 model on first launch
def download_re3_model():
    print("Downloading the re3 model.\nModel size is about 700mb so downloading might take a while.")

    dest_folder = './algorithms/re3/logs/'
    file = './algorithms/re3/logs/checkpoints.tar.gz'
    google_drive_id = '1mvxA9r9K1sydBEfVWk090f4Pdhg15YQD'

    gdd.download_file_from_google_drive(file_id=google_drive_id,
                                        dest_path=file,
                                        unzip=False)

    print("done downloading, unzipping model")
    tar = tarfile.open(file, "r:gz")
    tar.extractall(path=dest_folder)
    tar.close()

    print("done unzipping, deleting model checkpoints.tar.gz")

    if os.path.exists(file):
        os.remove(file)
        print("Successfully deleted checkpoints.tar.gz")
    else:
        print("Could not find zip to delete")
