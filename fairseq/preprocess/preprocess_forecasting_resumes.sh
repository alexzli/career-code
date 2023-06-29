# Get passed in arguments
#
# Usage: preprocess_forecasting_resumes.sh <resume_data_dir> <resume_binary_dir> <cutoff_year>
#
#
RESUME_DATA_DIR=$1
BINARY_DATA_DIR=$2
CUTOFF_YEAR=$3

echo "Thresholding resume data to cutoff year ($CUTOFF_YEAR)..."
python preprocess/convert_sequences_for_forecasting.py \
  --data-dir $RESUME_DATA_DIR \
  --dataset-name resumes \
  --cutoff-year $CUTOFF_YEAR
echo "...done."

# The test set contains the original test set.
cp $RESUME_DATA_DIR/test.job $RESUME_DATA_DIR/forecast_resumes/test.job   
cp $RESUME_DATA_DIR/test.year $RESUME_DATA_DIR/forecast_resumes/test.year   
cp $RESUME_DATA_DIR/test.education $RESUME_DATA_DIR/forecast_resumes/test.education   
cp $RESUME_DATA_DIR/test.location $RESUME_DATA_DIR/forecast_resumes/test.location

echo "Binarizing data..."

sh preprocess/preprocess_transfer_learning_resumes.sh \
  $RESUME_DATA_DIR/forecast_resumes \
  $BINARY_DATA_DIR
