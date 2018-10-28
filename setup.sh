# install pip
curl https://bootstrap.pypa.io/get-pip.py -o../get-pip.py
python ../get-pip.py --user

# setup virtualenv
export PATH=$PATH:/Users/ick/Library/Python/2.7/bin
pip install virtualenv --user
virtualenv venv 
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# setup directories
mkdir data
mkdir encoders
mkdir models
mkdir figures
mkdir stats
mkdir verbs

# download data
curl https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar > resized.tar
tar -xvf resized.tar
mv of500_images_resized resized_256
rm resized.tar

# preprocess training data
python preprocess_train.py imsitu_data/train.json gender > stats/gender_train_stats.txt
python preprocess_train.py imsitu_data/train.json fixed_gender_ratio --balanced_and_skewed > stats/fixed_gender_ratio_train_stats.txt
python preprocess_train.py imsitu_data/train.json activity_balanced --activity_balanced > stats/activity_balanced_train_stats.txt

# preprocess test data
python preprocess_test.py imsitu_data/test.json gender 
python preprocess_test.py imsitu_data/test.json activity_balanced 
python preprocess_test.py imsitu_data/test.json activity_balanced --men_only 
python preprocess_test.py imsitu_data/test.json activity_balanced --women_only 
python preprocess_test.py imsitu_data/test.json balanced_fixed_gender_ratio 
python preprocess_test.py imsitu_data/test.json skewed_fixed_gender_ratio 

# preprocess validation data
python preprocess_test.py imsitu_data/dev.json gender 
python preprocess_test.py imsitu_data/dev.json activity_balanced 
python preprocess_test.py imsitu_data/dev.json balanced_fixed_gender_ratio 
python preprocess_test.py imsitu_data/dev.json skewed_fixed_gender_ratio 
