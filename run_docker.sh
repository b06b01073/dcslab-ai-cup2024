current_dir=$(pwd)

docker build -t aicup_baseline .

docker run -it -v $current_dir:/root aicup_baseline

