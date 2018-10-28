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
mkdir model_output
mkdir models
mkdir figures
mkdir stats

# download data
curl https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar > resized.tar
tar -xvf resized.tar
mv of500_images_resized resized_256
rm resized.tar

# preprocess training data
python preprocess_train.py data/train.json data/genders_train.json data/human_verbs.txt > stats/gender_train_stats.txt
python preprocess_train.py data/train.json data/genders_train.json data/balanced_human_verbs.txt --balanced_and_skewed > stats/balanced_gender_train_stats.txt
python preprocess_train.py data/train.json data/activity_balanced_train.json data/activity_balanced_human_verbs.txt --activity_balanced > stats/activity_balanced_train_stats.txt

python preprocess_test.py data/human_verbs.txt data/test.json data/genders_test.json > stats/gender_test_stats.txt
python preprocess_test.py data/balanced_human_verbs.txt data/test.json data/skewed_genders_test.json > stats/skewed_gender_test_stats.txt
python preprocess_test.py data/balanced_human_verbs.txt data/test.json data/balanced_genders_test.json --balanced > stats/balanced_gender_test_stats.txt

python preprocess_test.py data/human_verbs.txt data/dev.json data/genders_dev.json > stats/gender_dev_stats.txt
python preprocess_test.py data/activity_balanced_human_verbs.txt data/dev.json data/activity_balanced_dev.json > stats/activity_balanced_dev_stats.txt
python preprocess_test.py data/balanced_human_verbs.txt data/dev.json data/balanced_genders_dev.json --balanced > stats/balanced_gender_dev_stats.txt
python preprocess_test.py data/balanced_human_verbs.txt data/test.json data/skewed_genders_dev.json > stats/skewed_gender_dev_stats.txt

python preprocess_test.py data/human_verbs.txt data/concat_test.json data/genders_concat_test.json > stats/gender_concat_test_stats.txt
python preprocess_test.py data/balanced_human_verbs.txt data/concat_test.json data/balanced_genders_concat_test.json --balanced > stats/balanced_gender_concat_test_stats.txt
