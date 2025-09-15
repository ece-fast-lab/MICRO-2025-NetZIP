git clone https://github.com/aliyun/SimAI.git
cd ./SimAI/

git submodule update --init --recursive
git submodule update --remote

sudo ./scripts/build.sh -c analytical
sudo ./scripts/build.sh -c ns3