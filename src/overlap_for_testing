def get_melspectrogram(wav_file_path, length=30, duration_of_segments=5, overlap=false, step_size=1):

    """
    Get mel spectrogram for a given wav file and divide it into parts.

    :param wav_file_path: Path to the source wav file
    :param length: length in seconds of the source audio file. Defaults to 30.
    :param duration_of_segments: duration of segments in seconds. number of segments = length/duration_of_segments. Defaults to 5.
    :return: Mel spectrogram of the source wav file. Segments will be saved to a file and it's path will be printed.
    """
    y, sr = librosa.load(wav_file_path, sr=None, duration=length)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Determine the number of samples in the duration of segments
    samples_per_segment = sr * duration_of_segments

    if overlap:
        samples_per_step = sr * step_size
        num_segments= length/step_size - duration_of_segments + 1
    else:
        num_segments = length/duration_of_segments

    # Loop through the audio signal and extract the segments
    for i in range(num_segments):
        # Get the start and end indices of the segment
        if overlap:
            start = i * samples_per_step
        else:
            start = i * samples_per_segment
        end = start + samples_per_segment

        # Extract the segment from the audio signal
        segment = y[start:end]

        # Compute the mel spectrogram of the segment
        mel_spec_segment = librosa.feature.melspectrogram(y=segment, sr=sr)

        # sample name
        sample_name = wav_file_path.replace("../data/genres_original/","").replace(".wav","")

        # path to save samples to
        sample_name = f'{sample_name.split("/")[0]}/npy/{sample_name.split("/")[1]}'

        # splitting files which contain overlapped vs distinct segments
        if overlap:
            directory = "overlap"
        else:
            directory = "distinct"


        save_path = f'../data/mel_spec_samples/{directory}/{sample_name}_{i}.npy'
        # Save the mel spectrogram to a file
        np.save(save_path, mel_spec_segment)
        print(f'Saved segment /{sample_name}_{i}.npy')

    return melspectrogram


def plot_melspectrogram(melspectrogram):
    """
    Plot mel spectrogram using pyplot to visualize the data. Works for both full spectrogram and segments.

    :param melspectrogram: np.array of mel spectrogram generated using librosa.feature.melspectrogram()
    :return: void
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()
