
import numpy as np


def decode_rle(annot_rle, shape):
    """
    Args : 
        annot_rle (str) : annotation in run-lenthg encoding
        shape (tuple) : (height, width) of an image

    Return : 
        mask (np.arr) : 1 - mask, 0 - background
        
    """
    s = annot_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
    for start, end in zip(starts, ends):
        mask[start:end] = 1
            
    return mask.reshape(shape)
    

def encode_rle(mask) : 
    """
    Args : 
        mask (np.arr) : 1 - mask, 0 - background

    Return : 
        annot_rle (str) : annotation in run-lenthg encoding
        
    """
    
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def build_masks(df_train, image_id, input_shape):
    """ Build masks from train annotations 
    Args : 
        dr_train (pd.Dataframe) : 

    Returns : 
        mask (np.arr)
    """
    height, width = input_shape
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
    mask = np.zeros((height, width))
    for label in labels:
        mask += decode_rle(label, shape=(height, width))
    mask = mask.clip(0, 1)
    return np.array(mask)
