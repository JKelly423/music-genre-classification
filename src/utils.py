


def get_melspectrogram(wav_file_path, length=30, duration_of_segments=5, overlap=False, duration_of_step=1):

    """
    Get mel spectrogram for a given wav file and divide it into parts.

    :param wav_file_path: Path to the source wav file
    :param length: length in seconds of the source audio file. Defaults to 30.
    :param duration_of_segments: duration of segments in seconds. number of segments = length/duration_of_segments. Defaults to 5.
    :param overlap: boolean determining whether slices of audio file will be overlapped or distinct. Defaults to False
    :param duration_of_step: step size from the beginning of one segment to the beginning of the next. Defaults to 1 second. Unused if overlap is False.
    :return: Mel spectrogram of the source wav file. Segments will be saved to a file and it's path will be printed.
    """
    y, sr = librosa.load(wav_file_path, sr=None, duration=length)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Determine the number of samples in the duration of segments
    samples_per_segment = sr * duration_of_segments

    if overlap:
        samples_per_step = sr * duration_of_step
        num_segments= length // duration_of_step - duration_of_segments + 1
    else:
        num_segments = length // duration_of_segments

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
        """ print(f'Saved segment /mel_spec_samples/{directory}/{sample_name}_{i}.npy') """

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

# Function to generate images from mel spectrograms for a given genre
def generate_images_for_genre(genre, num_samples=6, overlap=False):
    """
    Generate images from mel spectrogram segments for a given genre and save them to the same directory as the segments.

    :param genre: string of genre to generate images for
    :param num_samples: number of samples. Defaults to 6.
    :param overlap: boolean determining whether slices of audio file will be overlapped or distinct. Defaults to False. Needed here for directory purposes.
    :return: False if error, True if successful
    """
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    if genre not in genres:
        print(f'Genre \"{genre}\" not in list of genres')
        return False
    for j in range(0,100):
        sample_number = ""
        if j < 10:
            sample_number = f'0{j}'
        else:
            sample_number = f'{j}'

        # splitting files which contain overlapped vs distinct segments
        if overlap:
            directory = "overlap"
        else:
            directory = "distinct"

        for i in range(num_samples):
            mel_segment = np.load(f'../data/mel_spec_samples/{directory}/{genre}/npy/{genre}.000{sample_number}_{i}.npy')
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.power_to_db(mel_segment, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'{genre}.00000_{i}.npy')
            plt.tight_layout()
            save_name = f'../data/mel_spec_samples/{directory}/{genre}/png/{genre}.000{sample_number}_{i}.png'
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f'Saved image: {save_name}')

    return True


def get_data_and_labels(overlap=False):
    """
    Extracts data and labels from the mel spectrogram images and returns them as numpy arrays.

    :param overlap: boolean determining whether slices of audio file will be overlapped or distinct. Defaults to False. Needed here for directory purposes.
    :return: data and labels as numpy arrays
    """
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    data = []
    labels = []

    # splitting files which contain overlapped vs distinct segments
    if overlap:
        directory = "overlap"
    else:
        directory = "distinct"

    print(f'Extracting...')

    for genre in genres:
        print(f'Extracting {genre}')
        for file in os.listdir(f'../data/mel_spec_samples/{directory}/{genre}/npy'):

            # Extracting Mel Spectrogram feature
            # Use normalize_melspectrogram to get normalized mel spectrogram features to fit model
            melspectrogram = np.load(f'../data/mel_spec_samples/{directory}/{genre}/npy/{file}')

            if melspectrogram.shape[1] != 216:

                print("Bug")
                print(file)
                print(melspectrogram.shape[0])
                print(melspectrogram.shape[1])

            if melspectrogram.shape[0] != 128:

                print("Bug")
                print(file)
                print(melspectrogram.shape[0])
                print(melspectrogram.shape[1])

            # Extracting Label
            label = genres.index(genre)

            # Appending features and labels
            data.append(melspectrogram)
            labels.append(label)

    print('Finished extracting features and labels for all genres')
    return data, labels

def preprocess_mel_spectrogram(mel_spectrogram, normalize=True):
    """
    Preprocesses a mel spectrogram for use in a neural network.


    :param mel_spectrogram: (np.array) Mel spectrogram to be preprocessed.
    :param normalize: (bool) Whether to normalize the mel spectrogram. Defaults to True.


    :return mel_spectrogram: (np.array) Preprocessed mel spectrogram.
    """
    # Convert to decibels
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = np.float32(mel_spectrogram)

    # Normalize if specified
    if normalize:
        # Normalize to [0, 1]
        mel_spectrogram = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))

    # Convert to 3D array with an additional channel dimension
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)

    return mel_spectrogram


def train_test_split(x_data, y_data):
    """
    Splits data into training and testing sets.

    :param x_data: x data to be split
    :param y_data: y data to be split
    :return: x_train, y_train, x_test, y_test as numpy arrays
    """
    print(np.max(x_data))
    print(np.min(x_data))

    for i in range(0, len(x_data)):
        x_data[i] = preprocess_mel_spectrogram(x_data[i])

    x_shuffled, y_shuffled = shuffle(x_data, y_data, random_state=0)
    test_percentage = 0.2
    test_split = int(1-test_percentage * len(x_shuffled))
    x_test = x_shuffled[test_split:]
    y_test = y_shuffled[test_split:]
    #val_percentage = 0.2
    #val_split = int(1-test_percentage-val_percentage * len(x_shuffled))
    #x_val = x_shuffled[val_split:test_split]
    #y_val = y_shuffled[val_split:test_split]

    x_train = x_shuffled[:test_split]
    y_train = y_shuffled[:test_split]

    return x_train, y_train, x_test, y_test#, x_val, y_val