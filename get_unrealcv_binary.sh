Root=http://www.cs.jhu.edu/~qiuwch/animal
wget -c $Root/unrealcv_binary.zip -O ./data_generation/unrealcv_binary.zip
unzip ./data_generation/unrealcv_binary.zip -d ./data_generation/
rm -r ./data_generation/unrealcv_binary.zip
# download coco val 2017
wget -c http://images.cocodataset.org/zips/val2017.zip -O data_generation/val2017.zip
unzip ./data_generation/val2017.zip -d ./data_generation/
rm -r ./data_generation/val2017.zip
# install packages
pip install unrealcv
cd ./data_generation/unrealdb
pip install -e .
cd ../..
