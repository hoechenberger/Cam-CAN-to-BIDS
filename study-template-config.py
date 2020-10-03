import pathlib
from mne_bids.utils import get_entity_vals


study_name = 'Cam-CAN'
bids_root = pathlib.Path('/storage/store2/work/rhochenb/Data/Cam-CAN/BIDS')
subjects_dir = str(bids_root / 'derivatives' / 'freesurfer')
task = 'task'
# subjects_list = list(set(subjects_list) - set(exclude_subjects))
# subjects_list = 'all'

subjects_list = sorted(get_entity_vals(bids_root, entity_key='sub'))[:50]

ch_types = ['meg']
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_ctc_fname = pathlib.Path('/storage/store2/work/rhochenb/Data/Cam-CAN/'
                            'Cam-CAN_ct_sparse.fif')
mf_cal_fname = pathlib.Path('/storage/store2/work/rhochenb/Data/Cam-CAN/'
                            'Cam-CAN_sss_cal.dat')
tmin = -0.2
tmax = 0.5
baseline = (None, 0)
conditions = ['audiovis/300Hz', 'audiovis/600Hz', 'audiovis/1200Hz',
              'audiovis']

N_JOBS = 72
allow_maxshield = True
