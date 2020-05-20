# %%
import pathlib
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import mne
from mne_bids import write_raw_bids, make_bids_basename, write_anat

mne.set_log_level(verbose=False)

base_path = pathlib.Path('/storage/store/data/camcan/camcan47/cc700/meg/'
                         'pipeline/release004/data/aamod_meg_get_fif_00001')
bids_root = pathlib.Path('/storage/store2/work/rhochenb/Data/Cam-CAN/BIDS')
freesurfer_participants_dir = pathlib.Path('/storage/store/data/camcan-mne/'
                                           'freesurfer/')
# trans_dir = pathlib.Path('/storage/store/data/camcan-mne/trans/')
trans_dir = pathlib.Path('/storage/inria/agramfor/camcan_derivatives/'
                         'trans-krieger')

participants = sorted([p.parts[-1] for p in base_path.glob('*')])
experiments = ('rest', 'task', 'passive')

date_sound_card_change = datetime(month=12, day=8, year=2011,
                                  tzinfo=timezone.utc)

overview = pd.DataFrame(columns=['T1', 'trans', *experiments,
                                 'dataset_complete'],
                        index=pd.Index(participants, name='Participant'))
exclude = {'CC610462': ['task'],
           'CC512003': ['task'],
           'CC120208': ['passive'],
           'CC620685': ['passive'],
           'CC620044': ['rest']}  # Triggers missing

# restart_from = 'CC510534'
# restart_from = participants[-10]
restart_from = None

# %%
for participant in participants:
    t1w_fname = freesurfer_participants_dir / participant / 'mri' / 'T1.mgz'
    overview.loc[participant, 'T1'] = t1w_fname.exists()

    # trans_fname = trans_dir / f'sub-{participant}-trans.fif'
    # overview.loc[participant, 'trans'] = trans_fname.exists()

    try:
        trans_fname = list(trans_dir
                           .glob(f'{participant[2:]}-ve_tasks-??????.fif'))[0]
        trans_fname_exists = True
    except IndexError:
        trans_fname_exists = False

    overview.loc[participant, 'trans'] = trans_fname_exists

    for exp in experiments:
        raw_fname = base_path / participant / exp / f'{exp}_raw.fif'
        overview.loc[participant, exp] = raw_fname.exists()

    del trans_fname_exists, t1w_fname, raw_fname, exp

for participant, excluded in exclude.items():
    overview.loc[participant, excluded] = False

overview['dataset_complete'] = overview.iloc[:, :-1].all(axis='columns')


# %%
if restart_from is not None:
    overview = overview.loc[restart_from:, :]

event_name_to_id_mapping = {'audiovis/300Hz': 1,
                            'audiovis/600Hz': 2,
                            'audiovis/1200Hz': 3,
                            'catch/0': 4,
                            'catch/1': 5,
                            'audio/300Hz': 6,
                            'audio/600Hz': 7,
                            'audio/1200Hz': 8,
                            'vis/checker': 9}

event_id_to_name_mapping = dict(zip(event_name_to_id_mapping.values(),
                                    event_name_to_id_mapping.keys()))

event_name_to_duration_mapping = {'audiovis/300Hz': 0.3,
                                  'audiovis/600Hz': 0.3,
                                  'audiovis/1200Hz': 0.3,
                                  'catch/0': 0,
                                  'catch/1': 0,
                                  'audio/300Hz': 0.3,
                                  'audio/600Hz': 0.3,
                                  'audio/1200Hz': 0.3,
                                  'vis/checker': 0.034}

stim_chs = ('STI001', 'STI002', 'STI003', 'STI004')

# %%
print('Starting conversion.')
t_start = datetime.now()

for participant, dataset in overview.iterrows():
    msg = f'Participant {participant}: '

    if not dataset['dataset_complete']:
        msg += f'skipping (reason: dataset incomplete).'
        print(msg)
        continue

    participant_idx = overview.index.tolist().index(participant)
    percent_done = np.round((participant_idx / len(overview.index)) * 100, 1)
    msg += f'processing. [{percent_done:.1f}%]'
    print(msg)
    del participant_idx, percent_done

    for exp in experiments:
        print(f'… {exp}')
        raw_fname = base_path / participant / exp / f'{exp}_raw.fif'
        t1w_fname = (freesurfer_participants_dir / participant / 'mri' /
                     'T1.mgz')
        # trans_fname = trans_dir / f'sub-{participant}-trans.fif'
        trans_fname = list(trans_dir
                           .glob(f'{participant[2:]}-ve_tasks-??????.fif'))[0]

        raw = mne.io.read_raw_fif(raw_fname)
        # events = mne.find_events(raw,
        #                          stim_channel='STI101',
        #                          min_duration=0.002,
        #                          uint_cast=False,
        #                          initial_event=True)

        # Work around an acquisition bug in STI101: construct a stimulus
        # channel ourselves from STI001:STI004.
        stim_data = (raw
                     .copy()
                     .load_data()
                     .pick_channels(stim_chs)
                     .get_data())
        stim_data /= 5  # Signal is always +5V

        # First channel codes for bit 1, second for bit 2, etc.
        for stim_ch_idx, _ in enumerate(stim_chs):
            stim_data[stim_ch_idx, :] *= 2**stim_ch_idx

        # Create a virtual channel which is the sum of the individual channels.
        stim_data = stim_data.sum(axis=0, keepdims=True)
        info = mne.create_info(['STI_VIRTUAL'], raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info, first_samp=raw.first_samp)

        events = mne.find_events(stim_raw, stim_channel='STI_VIRTUAL',
                                 min_duration=0.002)

        del stim_data, stim_raw, info

        if exp != 'rest':
            print(f'Found events: {sorted(set(events[:, 2]))}')

            # Create Annotations
            onsets = []
            durations = []
            descriptions = []
            sfreq = raw.info['sfreq']

            before_sound_card = date_sound_card_change >= raw.info['meas_date']

            scdelay = 13 if before_sound_card else 26

            for event in events:
                onset_sample = event[0]
                onset = onset_sample / sfreq
                description = event_id_to_name_mapping[event[2]]
                duration = event_name_to_duration_mapping[description]

                # Apply delays in stimuli following file get_trial_info.m
                # in the Cam-CAN release file
                if event[2] in [6, 7, 8, 774, 775, 776]:
                    assert exp == 'passive'
                    delay = scdelay  # audio delay
                elif event[2] in [9, 777]:
                    assert exp == 'passive'
                    delay = 34  # visual delay
                elif event[2] in [1, 2, 3]:
                    assert exp == 'task'
                    delay = (scdelay + 34) // 2  # take mean between audio and vis
                elif event[2] in [4, 5]:
                    pass  # catch have no delay
                else:
                    raise ValueError('Trigger not found')

                onset += delay / sfreq

                onsets.append(onset)
                descriptions.append(description)
                durations.append(duration)

                del onset_sample, onset, duration, description

            annotations = mne.Annotations(onset=onsets,
                                          duration=durations,
                                          description=descriptions,
                                          orig_time=raw.info['meas_date'])
            raw.set_annotations(annotations)
            del annotations, onsets, descriptions, durations

        bids_basename = make_bids_basename(subject=participant, task=exp)
        write_raw_bids(raw, bids_basename,
                       bids_root=bids_root,
                       events_data=events,
                       event_id=event_name_to_id_mapping,
                       overwrite=True,
                       verbose=False)

        write_anat(bids_root=bids_root,
                   subject=participant,
                   t1w=t1w_fname,
                   acquisition='t1w',
                   trans=trans_fname,
                   raw=raw,
                   overwrite=True,
                   verbose=False)

        del bids_basename, trans_fname, raw_fname, raw, events


print('Finished conversion.')
t_end = datetime.now()

print(f'Process took {t_end - t_start}.')
