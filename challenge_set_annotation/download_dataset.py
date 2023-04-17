from datasets import load_dataset
import argparse, os

def download_dataset(out_path):
    dataset = load_dataset("nikitam/ACES")
    dataset.save_to_disk(out_path)
    print("saved to ", out_path)
    
if __name__ == "__main__":
    # Get arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-o", "--out-path", help="path to the folder for the dataset")
    # args = parser.parse_args()
    
    folder = os.getcwd()
    dataset_path = os.path.join(folder, 'dataset')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        download_dataset(dataset_path)
    else:
        print('{} already exists, skipping..\n'.format(dataset_path))
