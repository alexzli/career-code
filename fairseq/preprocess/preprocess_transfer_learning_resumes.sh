RESUME_DATA_DIR=$1
BINARY_DATA_DIR=$2


RESUME_SUFFIX="forecast-resumes"

echo "Combining resume data..."
cat $RESUME_DATA_DIR/train.job $RESUME_DATA_DIR/valid.job \
    $RESUME_DATA_DIR/test.job > $RESUME_DATA_DIR/all.job

cat $RESUME_DATA_DIR/train.year $RESUME_DATA_DIR/valid.year \
    $RESUME_DATA_DIR/test.year > $RESUME_DATA_DIR/all.year

cat $RESUME_DATA_DIR/train.education $RESUME_DATA_DIR/valid.education \
    $RESUME_DATA_DIR/test.education > $RESUME_DATA_DIR/all.education

cat $RESUME_DATA_DIR/train.location $RESUME_DATA_DIR/valid.location \
    $RESUME_DATA_DIR/test.location > $RESUME_DATA_DIR/all.location
echo "...done."

## Create dictionaries from combined data.
echo "Creating dictionary for jobs..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/all.job \
    --validpref $RESUME_DATA_DIR/valid.job \
    --testpref $RESUME_DATA_DIR/test.job \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/job \
    --dict-only \
    --workers 60
echo "...done."

echo "Creating dictionary for years..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/all.year \
    --validpref $RESUME_DATA_DIR/valid.year \
    --testpref $RESUME_DATA_DIR/test.year \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/year \
    --dict-only \
    --workers 60
echo "...done."

echo "Creating dictionary for educations..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/all.education \
    --validpref $RESUME_DATA_DIR/valid.education \
    --testpref $RESUME_DATA_DIR/test.education \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/education \
    --dict-only \
    --workers 60
echo "...done."

echo "Creating dictionary for locations..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/all.location \
    --validpref $RESUME_DATA_DIR/valid.location \
    --testpref $RESUME_DATA_DIR/test.location \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/location \
    --dict-only \
    --workers 60
echo "...done."

## Modify year dictionary so years are ordered (this is helpful for
## forecasting).
python preprocess/create_ordered_year_dictionary.py \
    --data-dir $BINARY_DATA_DIR/$RESUME_SUFFIX/year

## Preprocess resume data using the combined dictionaries.
echo "Preprocessing resume data (jobs)..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.job \
    --validpref $RESUME_DATA_DIR/valid.job \
    --testpref $RESUME_DATA_DIR/test.job \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/job \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/job/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing resume data (years)..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.year \
    --validpref $RESUME_DATA_DIR/valid.year \
    --testpref $RESUME_DATA_DIR/test.year \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/year \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/year/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing resume data (educations)..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.education \
    --validpref $RESUME_DATA_DIR/valid.education \
    --testpref $RESUME_DATA_DIR/test.education \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/education \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/education/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing resume data (locations)..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.location \
    --validpref $RESUME_DATA_DIR/valid.location \
    --testpref $RESUME_DATA_DIR/test.location \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/location \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/location/dict.txt \
    --workers 60
echo "...done."
