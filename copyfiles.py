import shutil
import os
import argparse
from tqdm import tqdm

def copy_files_and_dirs(source_dir, destination_dir):
    """Copies all files and subdirectories from source_dir to destination_dir with progress tracking.

    Args:
        source_dir: Path to the source directory.
        destination_dir: Path to the destination directory.
    """
    try:
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source '{source_dir}' not found.")

        if os.path.exists(destination_dir):
            if not os.path.isdir(destination_dir):
                raise NotADirectoryError(f"Destination '{destination_dir}' is not a directory")
            
            print(f"Destination '{destination_dir}' already exists. Merging contents...")

            # Get total number of files and directories for progress tracking
            all_items = list(os.scandir(source_dir))

            with tqdm(total=len(all_items), desc="Copying", unit="item") as pbar:
                for item in all_items:
                    src_path = item.path
                    dest_path = os.path.join(destination_dir, item.name)

                    if item.is_dir():
                        if not os.path.exists(dest_path):  # Avoid `copytree` error when dest exists
                            shutil.copytree(src_path, dest_path)
                        else:
                            print(f"Skipping existing directory: {dest_path}")
                    else:
                        shutil.copy2(src_path, dest_path)

                    pbar.update(1)  # Update progress bar

        else:
            print(f"Copying entire directory '{source_dir}' to '{destination_dir}'...")
            all_items = list(os.scandir(source_dir))

            with tqdm(total=len(all_items), desc="Copying", unit="item") as pbar:
                shutil.copytree(source_dir, destination_dir)  # Works when destination does not exist
                pbar.update(len(all_items))  # Mark as completed

            print(f"Contents of '{source_dir}' copied successfully to '{destination_dir}'.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except NotADirectoryError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy files and directories.")
    parser.add_argument("--source_dir", default="G:\\CochlScene\\audio", help="Path to the source directory.")
    parser.add_argument("--destination_dir", default="F:\\cochl_backup", help="Path to the destination directory.")
    
    args = parser.parse_args()

    copy_files_and_dirs(args.source_dir, args.destination_dir)
