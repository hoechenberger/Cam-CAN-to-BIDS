#!/bin/zsh

MNE_BIDS_STUDY_CONFIG="/home/parietal/rhochenb/Development/Cam-CAN/study-template-config.py"
BIDS_ROOT="/storage/store2/work/rhochenb/Data/Cam-CAN/BIDS"

STUDY_TEMPLATE_DIR="~/Development/mne-study-template"

# make sensor source report
MNE_BIDS_STUDY_CONFIG=$MNE_BIDS_STUDY_CONFIG make -C $STUDY_TEMPLATE_DIR sensor reports