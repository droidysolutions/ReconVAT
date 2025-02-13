from model.utils import convert_wav_to_flac,convert_xml_to_tsv,delete_wav_xml_files

convert_wav_to_flac('./IDMT-SMT-GUITAR_V2')

convert_xml_to_tsv('./IDMT-SMT-GUITAR_V2')

delete_wav_xml_files('./IDMT-SMT-GUITAR_V2')

import os
import glob

def available_groups():
        return ['dataset1', 'dataset2', 'dataset3','dataset4']

def files(group):
    """Return zip(flacs, tsvs) only for properly paired (flac, tsv) files."""
    
    assert group in available_groups(), f"❌ Invalid dataset group: {group}"

    # Base dataset path (where dataset1, dataset2, dataset3 are stored)
    dataset_path = os.path.join('./IDMT-SMT-GUITAR_V2', group)

    # Find all .flac and .tsv files recursively in subdirectories
    flacs = sorted(glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True))
    tsvs = sorted(glob.glob(os.path.join(dataset_path, "**", "*.tsv"), recursive=True))

    # Debugging output
    print(f"🔍 Searching in: {dataset_path}")
    print(f"🎵 Found {len(flacs)} audio files (.flac)")
    print(f"📜 Found {len(tsvs)} annotation files (.tsv)")

    # Dictionary for fast lookup
   # tsv_set = set(tsvs)
    print(f"✅ tsv_set $$$$$$$$$$$$$  $$$$$$$$$$$$")
    
    # Ensure only (flac, tsv) pairs that match
    paired_flacs = []
    paired_tsvs = []

    for flac in flacs:
        flac_name = os.path.splitext(os.path.basename(flac))[0]  # Get filename without extension
        #tsv = flac.replace("audio", f"annotation").replace(".flac",".tsv")  # Expected tsv path
        tsv = flac.replace(os.path.sep + "audio" + os.path.sep, os.path.sep + "annotation" + os.path.sep).replace(".flac", ".tsv")
        # print(f"✅ flac_name {flac_name} 0000000000000")
        # print(f"✅ Path {flac} ------- {tsv}")
        if tsv in tsvs:
            paired_flacs.append(flac)
            paired_tsvs.append(tsv)

    print(f"✅ Successfully paired {len(paired_flacs)} (flac, tsv) files.")

    return zip(paired_flacs, paired_tsvs)  # ✅ Return only correctly paired files


def exe():

    dataset_path1 = os.path.join('./IDMT-SMT-GUITAR_V2', 'dataset1')
    dataset_path2 = os.path.join('./IDMT-SMT-GUITAR_V2', 'dataset2')
    dataset_path3 = os.path.join('./IDMT-SMT-GUITAR_V2', 'dataset3')
    dataset_path4 = os.path.join('./IDMT-SMT-GUITAR_V2', 'dataset4')

    # Find all .flac and .tsv files recursively in subdirectories
    flacs1 = sorted(glob.glob(os.path.join(dataset_path1, "**", "*.flac"), recursive=True))
    flacs2 = sorted(glob.glob(os.path.join(dataset_path2, "**", "*.flac"), recursive=True))
    flacs3 = sorted(glob.glob(os.path.join(dataset_path3, "**", "*.flac"), recursive=True))
    flacs4 = sorted(glob.glob(os.path.join(dataset_path4, "**", "*.flac"), recursive=True))
    tsvs = sorted(glob.glob(os.path.join(dataset_path1, "**", "*.tsv"), recursive=True))

    # Debugging output
    #print(f"🔍 Searching in: {dataset_path}")
    print(f"🎵 Found {len(flacs1+flacs2+flacs3+flacs4)} audio files(.flac)")
   # print(f"📜 Found {len(tsvs)} annotation files (.tsv)")

    # paired_files = files('dataset4')
    # for index, (flac, tsv) in enumerate(paired_files):
    #     print(f"{index}: 🎵 Audio: {flac} -> 📜 Annotation: {tsv}")



exe()