# install pip
curl https://bootstrap.pypa.io/get-pip.py -o ~/Downloads/get-pip.py
python ~/Downloads/get-pip.py --user

# setup virtualenv
export PATH=$PATH:~/Library/Python/2.7/bin
pip install virtualenv --user
virtualenv venv 
source venv/bin/activate

# install dependencies
pip install -r requirements.txt --user

# download data
./install.sh

#setup directories
mkdir model_output

# {u'frames': [{u'item': u'n07678729', u'destination': u'n03619890', u'place': u'n01794158', u'agent': u'n10787470'}], u'verb': u'stuffing'}
