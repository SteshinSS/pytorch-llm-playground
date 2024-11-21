import urllib.request

verdict_url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
file_path = "data/the-verdict.txt"
urllib.request.urlretrieve(verdict_url, file_path)
