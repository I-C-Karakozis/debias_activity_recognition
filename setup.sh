# install pip
curl https://bootstrap.pypa.io/get-pip.py -o ~/Downloads/get-pip.py
python ~/Downloads/get-pip.py

# setup virtualenv
export PATH=$PATH:/Users/ick/Library/Python/2.7/bin
pip install virtualenv
virtualenv venv 
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# download data
./install.sh
rm resized.tar

#setup directories
mkdir model_output

# {u'frames': [{u'item': u'n07678729', u'destination': u'n03619890', u'place': u'n01794158', u'agent': u'n10787470'}], u'verb': u'stuffing'}
