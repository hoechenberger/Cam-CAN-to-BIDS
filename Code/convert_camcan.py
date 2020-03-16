"""Convert Cam-CAN dataset to BIDS format"""

import glob
import os.path as op

import mne
from mne_bids import write_raw_bids, make_bids_basename, write_anat

base_path = '/storage/store/data/camcan/camcan47/cc700/meg/pipeline/release004/data/aamod_meg_get_fif_00001'
subjects = list(map(op.basename, sorted(glob.glob(base_path + '/*'))))
tasks = ['rest', 'task']
bids_root = '/storage/store/data/camcan-bids'
subjects_dir = '/storage/store/data/camcan-mne/freesurfer/'
trans_dir = '/storage/store/data/camcan-mne/trans/'

# exclude = ['CC120120']
# # exclude += ['CC120264']  # event duration problem
# exclude += ['CC221040']  # no task data
# exclude += ['CC221054']  # no rest and task data
excluded = []

# for subject in ['CC120120']:
for subject in subjects:
    # if subject in exclude:
    #     continue
    t1w = subjects_dir + "/%s/mri/T1.mgz" % subject
    trans = trans_dir + '/sub-%s-trans.fif' % subject

    # if not op.exists(trans):
    #     continue

    try:
        # Take care of MEG
        for task in tasks:
            raw_fname = op.join(base_path, subject, task, '%s_raw.fif' % task)
            raw = mne.io.read_raw_fif(raw_fname)
            bids_basename = make_bids_basename(subject=subject, task=task)
            events = mne.find_events(raw, min_duration=0.002, initial_event=True)
            write_raw_bids(raw, bids_basename,
                           output_path=bids_root,
                           events_data=events,
                           overwrite=True)

        # Take care of anatomy
        write_anat(bids_root, subject, t1w, acquisition="t1w",
                   trans=trans, raw=raw, overwrite=True)        
    except Exception as e:
        excluded.append(subject)
