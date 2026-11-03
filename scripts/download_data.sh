set -e
mkdir -p data/rice data/cuhkcr
kaggle datasets download -d shubhank001/rice-remote-sensing-images-for-cloud-removal -p data/rice --unzip
huggingface-cli download littlebeen/DE C-CUHK.rar --local-dir data/cuhkcr
cd data/cuhkcr && (unrar x -y C-CUHK.rar || 7z x -y C-CUHK.rar)
