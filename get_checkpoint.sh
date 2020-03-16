Root=http://www.cs.jhu.edu/~qiuwch/animal/
mkdir -p checkpoint/real_animal checkpoint/synthetic_animal/ 
wget -c $Root/checkpoint/real_animal/horse.zip -O ./checkpoint/real_animal/horse.zip
wget -c $Root/checkpoint/real_animal/tiger.zip -O ./checkpoint/real_animal/tiger.zip
unzip ./checkpoint/real_animal/horse.zip -d ./checkpoint/real_animal/
unzip ./checkpoint/real_animal/tiger.zip -d ./checkpoint/real_animal/
rm -r ./checkpoint/real_animal/horse.zip
rm -r ./checkpoint/real_animal/tiger.zip
wget -c $Root/checkpoint/synthetic_animal/horse.zip -O ./checkpoint/synthetic_animal/horse.zip
wget -c $Root/checkpoint/synthetic_animal/tiger.zip -O ./checkpoint/synthetic_animal/tiger.zip
unzip ./checkpoint/synthetic_animal/horse.zip -d ./checkpoint/synthetic_animal/
unzip ./checkpoint/synthetic_animal/tiger.zip -d ./checkpoint/synthetic_animal/
rm -r ./checkpoint/synthetic_animal/horse.zip
rm -r ./checkpoint/synthetic_animal/tiger.zip
