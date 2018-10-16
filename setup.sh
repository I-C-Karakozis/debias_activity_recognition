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

# setup directories
mkdir model_output

# download data
curl https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar > resized.tar
tar -xvf resized.tar
mv of500_images_resized resized_256

mkdir baseline_models
curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_encoder > baseline_models/baseline_encoder
curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_resnet_101 > baseline_models/baseline_resnet_101
curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_resnet_50 > baseline_models/baseline_resnet_50
curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_resnet_34 > baseline_models/baseline_resnet_34

rm resized.tar

# preprocess data
python preprocess_train.py data/train.json data/genders_train.json data/human_verbs.txt
python preprocess_test.py data/human_verbs.txt data/test.json data/genders_test.json 
python preprocess_test.py data/human_verbs.txt data/dev.json data/genders_dev.json

# {u'frames': [{u'item': u'n07678729', u'destination': u'n03619890', u'place': u'n01794158', u'agent': u'n10787470'}], u'verb': u'stuffing'}
